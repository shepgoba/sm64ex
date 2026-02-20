#ifdef RAPI_METAL
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <format>
#include <iomanip>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


#include "gfx_rendering_api.h"
#include "gfx_cc.h"
#include "../configfile.h"

#define DECLARE_GFX_SDL_FUNCTIONS
#include "gfx_sdl.h"

void log_event(const char *fmt, ...) {
    return;
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_struct = *std::localtime(&t);

    auto duration = now.time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 1000000;

    std::cout << "[" << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S");
    std::cout << "." << std::setfill('0') << std::setw(6) << microseconds << "] ";

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

struct TextureDataMetal {
    MTL::Texture *texture = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ShaderProgramMetal {
    std::string source; // debugging
    uint32_t shader_id;
    uint8_t num_inputs;
    bool used_textures[2];
    uint8_t num_floats;

    MTL::Library *library = nullptr;
    MTL::VertexDescriptor *vertex_descriptor = nullptr;
	MTL::RenderPipelineState *pipeline = nullptr;
};

struct {
    CA::MetalLayer *layer = nullptr;
    MTL::Device *device = nullptr;
    MTL::CommandQueue *queue = nullptr;

    MTL::Viewport viewport;

    struct ShaderProgramMetal shader_program_pool[64];
    uint8_t shader_program_pool_size;

    ShaderProgramMetal *active_shader = nullptr;

	std::vector<TextureDataMetal> textures;

    uint32_t current_texture_ids[2] = {};
    int current_tile = 0;

	// --- Per-frame state ---
    CA::MetalDrawable *current_drawable = nullptr;
    MTL::CommandBuffer *current_cmd_buffer = nullptr;
    MTL::RenderCommandEncoder *current_encoder = nullptr;
    MTL::RenderPassDescriptor *current_pass_desc = nullptr;

	MTL::Buffer *dynamic_vertex_buffer = nullptr;
    size_t dynamic_offset = 0;

} mtl_state;


static const char *shader_item_to_str(uint32_t item, bool with_alpha, bool only_alpha, bool inputs_have_alpha, bool hint_single_element) {
    if (!only_alpha) {
        switch (item) {
            default:
            case SHADER_0:
                return with_alpha ? "float4(0.0, 0.0, 0.0, 0.0)" : "float3(0.0, 0.0, 0.0)";
            case SHADER_INPUT_1:
                return with_alpha || !inputs_have_alpha ? "in.input1" : "in.input1.rgb";
            case SHADER_INPUT_2:
                return with_alpha || !inputs_have_alpha ? "in.input2" : "in.input2.rgb";
            case SHADER_INPUT_3:
                return with_alpha || !inputs_have_alpha ? "in.input3" : "in.input3.rgb";
            case SHADER_INPUT_4:
                return with_alpha || !inputs_have_alpha ? "in.input4" : "in.input4.rgb";
            case SHADER_TEXEL0:
                return with_alpha ? "texVal0" : "texVal0.rgb";
            case SHADER_TEXEL0A:
                return hint_single_element ? "texVal0.a" : (with_alpha ? "float4(texVal0.a, texVal0.a, texVal0.a, texVal0.a)" : "float3(texVal0.a, texVal0.a, texVal0.a)");
            case SHADER_TEXEL1:
                return with_alpha ? "texVal1" : "texVal1.rgb";
        }
    } else {
        switch (item) {
            default:
            case SHADER_0:
                return "0.0";
            case SHADER_INPUT_1:
                return "input.input1.a";
            case SHADER_INPUT_2:
                return "input.input2.a";
            case SHADER_INPUT_3:
                return "input.input3.a";
            case SHADER_INPUT_4:
                return "input.input4.a";
            case SHADER_TEXEL0:
                return "texVal0.a";
            case SHADER_TEXEL0A:
                return "texVal0.a";
            case SHADER_TEXEL1:
                return "texVal1.a";
        }
    }
}


static std::string generate_formula(const uint8_t c[2][4],
                                    bool do_single, bool do_multiply, bool do_mix,
                                    bool with_alpha, bool only_alpha, bool inputs_have_alpha) {
    std::string expr;

    auto item = [&](int index, bool hint_single=false) {
        return shader_item_to_str(c[only_alpha][index], with_alpha, only_alpha, inputs_have_alpha, hint_single);
    };

    if (do_single) {
        expr += item(3);
    } else if (do_multiply) {
        expr += item(0);
        expr += " * ";
        expr += item(2,true);
    } else if (do_mix) {
        expr += "mix(";
        expr += item(1);
        expr += ", ";
        expr += item(0);
        expr += ", ";
        expr += item(2,true);
        expr += ")";
    } else {
        expr += "(";
        expr += item(0);
        expr += " - ";
        expr += item(1);
        expr += ") * ";
        expr += item(2,true);
        expr += " + ";
        expr += item(3);
    }

    return expr;
}

static bool gfx_metal_z_is_from_0_to_1(void) {
    return true;
}

static void gfx_metal_unload_shader(ShaderProgram *old_prg) {
    (void)old_prg;
}

static void gfx_metal_load_shader(ShaderProgram *new_prg) {
    //printf("moving to new shader: %p\n", new_prg);
    mtl_state.active_shader = reinterpret_cast<ShaderProgramMetal *>(new_prg);
}

std::ostream &operator<<(std::ostream &stream, const CCFeatures &cc) {
    stream << "CCFeatures {\n";

    stream << "  c:\n";
    for (int i = 0; i < 2; ++i) {
        stream << "    [" << i << "] { ";
        for (int j = 0; j < 4; ++j) {
            stream << static_cast<int>(cc.c[i][j]);
            if (j != 3) stream << ", ";
        }
        stream << " }\n";
    }

    stream << "  opt_alpha: " << cc.opt_alpha << "\n";
    stream << "  opt_fog: " << cc.opt_fog << "\n";
    stream << "  opt_texture_edge: " << cc.opt_texture_edge << "\n";
    stream << "  opt_noise: " << cc.opt_noise << "\n";

    stream << "  used_textures: { "
           << cc.used_textures[0] << ", "
           << cc.used_textures[1] << " }\n";

    stream << "  num_inputs: " << cc.num_inputs << "\n";

    stream << "  do_single: { "
           << cc.do_single[0] << ", "
           << cc.do_single[1] << " }\n";

    stream << "  do_multiply: { "
           << cc.do_multiply[0] << ", "
           << cc.do_multiply[1] << " }\n";

    stream << "  do_mix: { "
           << cc.do_mix[0] << ", "
           << cc.do_mix[1] << " }\n";

    stream << "  color_alpha_same: " << cc.color_alpha_same << "\n";

    stream << "}";

    return stream;
}


void gfx_metal_load_shader(ShaderProgram *);
static ShaderProgram *gfx_metal_create_and_load_new_shader(uint32_t shader_id) {
    CCFeatures ccf;
    gfx_cc_get_features(shader_id, &ccf);

    auto vertexDescriptor = MTL::VertexDescriptor::vertexDescriptor();

    std::string vertex_shader_source = "";
    vertex_shader_source += "#include <metal_stdlib>\nusing namespace metal;\n";

    std::string input_struct_source = "struct VertexIn {\n";
    std::string output_struct_source = "struct VertexOut {\n";

    input_struct_source += "float4 pos [[attribute(0)]];\n";
    output_struct_source += "float4 pos [[position]];\n";
    vertexDescriptor->attributes()->object(0)->setFormat(MTL::VertexFormatFloat4);
    vertexDescriptor->attributes()->object(0)->setOffset(0);
    vertexDescriptor->attributes()->object(0)->setBufferIndex(0);


    int input_idx = 1;
    int shader_input_idx = 1;
    int shader_output_idx = 1;
    float num_floats = 0;

    num_floats += 4;

    if (ccf.used_textures[0] || ccf.used_textures[1]) {
        input_struct_source += std::format("float2 tex_coord [[attribute({})]];\n", 
            shader_input_idx);
        output_struct_source += std::format("float2 tex_coord;\n");

        vertexDescriptor->attributes()->object(shader_input_idx)->setFormat(MTL::VertexFormatFloat2);
        vertexDescriptor->attributes()->object(shader_input_idx)->setOffset(num_floats * sizeof(float));
        vertexDescriptor->attributes()->object(shader_input_idx)->setBufferIndex(0);

        shader_output_idx++;
        num_floats += 2;
        shader_input_idx++;
    }

    if (ccf.opt_fog) {
        input_struct_source += std::format("float4 fog [[attribute({})]];\n", 
            shader_input_idx);
        output_struct_source += std::format("float4 fog;\n");
        vertexDescriptor->attributes()->object(shader_input_idx)->setFormat(MTL::VertexFormatFloat4);
        vertexDescriptor->attributes()->object(shader_input_idx)->setOffset(num_floats * sizeof(float));
        vertexDescriptor->attributes()->object(shader_input_idx)->setBufferIndex(0);
        shader_input_idx++;
        shader_output_idx++;
        num_floats += 4;
    }

    for (int i = 0; i < ccf.num_inputs; i++) {
        int input_n_floats = ccf.opt_alpha ? 4 : 3;
        input_struct_source += std::format("float{} input{} [[attribute({})]];\n", input_n_floats, input_idx, shader_input_idx);
        output_struct_source += std::format("float{} input{};\n", input_n_floats, input_idx);
        vertexDescriptor->attributes()->object(shader_input_idx)->setFormat(input_n_floats == 4 ? MTL::VertexFormatFloat4 : MTL::VertexFormatFloat3);
        vertexDescriptor->attributes()->object(shader_input_idx)->setOffset(num_floats * sizeof(float));
        vertexDescriptor->attributes()->object(shader_input_idx)->setBufferIndex(0);
        shader_input_idx++;
        shader_output_idx++;
        input_idx++;
        num_floats += input_n_floats;
    }
    input_struct_source += "};\n\n";
    output_struct_source += "};\n\n";
    vertex_shader_source += input_struct_source;
    vertex_shader_source += output_struct_source;

    vertex_shader_source += R"(vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
VertexOut out;
out.pos = in.pos;
)";

    if (ccf.used_textures[0] || ccf.used_textures[1]) {
        vertex_shader_source += "out.tex_coord = in.tex_coord;\n";
    }
    if (ccf.opt_fog) {
        vertex_shader_source += "out.fog = in.fog;\n";
    }
    for (int i = 0; i < ccf.num_inputs; i++) {
        vertex_shader_source += std::format("out.input{} = in.input{};\n", input_idx - i - 1, input_idx - i - 1);
    }

    vertex_shader_source += "return out;\n}\n\n";

    std::string fs;

    // Header
    fs += "#include <metal_stdlib>\n";
    fs += "using namespace metal;\n\n";


    // Uniforms
    fs += "struct FragmentUniforms {\n";
    if (ccf.used_textures[0]) fs += "    float2 uTex0Size;\n    uint uTex0Filter;\n";
    if (ccf.used_textures[1]) fs += "    float2 uTex1Size;\n    uint uTex1Filter;\n";
    if (ccf.opt_alpha && ccf.opt_noise) fs += "    float frame_count;\n";
    fs += "};\n\n";

    // Sampling helper
    if (ccf.used_textures[0] || ccf.used_textures[1]) {
        fs += "float4 sampleTex(texture2d<float> tex, sampler samp, float2 uv, float2 texSize, uint dofilter) {\n";
        if (configFiltering == 2) {
            fs += "    float2 offset = fract(uv * texSize - float2(0.5));\n";
            fs += "    offset -= step(1.0, offset.x + offset.y);\n";
            fs += "    float2 invSize = 1.0 / texSize;\n";
            fs += "    float4 c0 = tex.sample(samp, uv - offset * invSize);\n";
            fs += "    float4 c1 = tex.sample(samp, uv - float2(offset.x - sign(offset.x), offset.y) * invSize);\n";
            fs += "    float4 c2 = tex.sample(samp, uv - float2(offset.x, offset.y - sign(offset.y)) * invSize);\n";
            fs += "    return c0 + abs(offset.x)*(c1-c0) + abs(offset.y)*(c2-c0);\n";
        } else {
            fs += "    return tex.sample(samp, uv);\n";
        }
        fs += "}\n\n";
    }

    // Random function
    if (ccf.opt_alpha && ccf.opt_noise) {
        fs += "float random(float3 value) {\n";
        fs += "    float r = dot(sin(value), float3(12.9898, 78.233, 37.719));\n";
        fs += "    return fract(sin(r) * 143758.5453);\n";
        fs += "}\n\n";
    }

    // Fragment function
    fs += "fragment float4 fragment_main(\n";
    fs += "    VertexOut in [[stage_in]],\n";
    fs += "    constant FragmentUniforms& uniforms [[buffer(0)]]\n";
    if (ccf.used_textures[0] || ccf.used_textures[1]) {
        fs += ",";
    }
    if (ccf.used_textures[0]) {
        fs += "    texture2d<float> uTex0 [[texture(0)]], sampler samp0 [[sampler(0)]]";
        if (ccf.used_textures[1]) fs+= ',';
        fs+= '\n';
     }
    if (ccf.used_textures[1]) fs += "    texture2d<float> uTex1 [[texture(1)]], sampler samp1 [[sampler(1)]]\n";
    fs += ") {\n";

    // Sample textures
    if (ccf.used_textures[0]) fs += "    float4 texVal0 = sampleTex(uTex0, samp0, in.tex_coord, uniforms.uTex0Size, uniforms.uTex0Filter);\n";
    if (ccf.used_textures[1]) fs += "    float4 texVal1 = sampleTex(uTex1, samp1, in.tex_coord, uniforms.uTex1Size, uniforms.uTex1Filter);\n";

    std::string formula = generate_formula(ccf.c, ccf.do_single[0], ccf.do_multiply[0], ccf.do_mix[0], ccf.opt_alpha, false, ccf.opt_alpha);
    fs += ccf.opt_alpha ? "float4" : "float3";
    fs += " texel = " + formula + ";\n";

    // Edge alpha
    if (ccf.opt_texture_edge && ccf.opt_alpha)
        fs += "    if (texel.a <= 0.3) discard_fragment(); else texel.a = 1.0;\n";

    // Fog
    if (ccf.opt_fog) {
        if (ccf.opt_alpha) fs += "    texel.rgb = mix(texel.rgb, in.fog.rgb, in.fog.a);\n";
        else fs += "    texel = mix(texel, in.fog.rgb, in.fog.a);\n";
    }

    // Noise alpha
    if (ccf.opt_alpha && ccf.opt_noise)
        fs += "    texel.a *= floor(random(float3(floor(in.pos.xy), uniforms.frame_count)) + 0.5);\n";

    // Return
    fs += ccf.opt_alpha ? "    return texel;\n" : "    return float4(texel, 1.0);\n";

    fs += "}\n";


    std::string shader_combined_source = vertex_shader_source + fs;
   // std::string shader_name = "mtlShader" + std::to_string(shader_id);
    std::cout << shader_combined_source << '\n';
    NS::Error *error = nullptr;
    auto library = mtl_state.device->newLibrary(
        NS::String::string(shader_combined_source.c_str(), NS::StringEncoding::UTF8StringEncoding), nullptr, &error);
    if (!library) {
        std::cout << "Library compilation failed: " << error->localizedDescription()->utf8String() << std::endl;
        exit(-1);
    }

    vertexDescriptor->layouts()->object(0)->setStride(num_floats * sizeof(float));
    vertexDescriptor->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);
    vertexDescriptor->layouts()->object(0)->setStepRate(1);

	ShaderProgramMetal *prg = &mtl_state.shader_program_pool[mtl_state.shader_program_pool_size++];
    prg->shader_id = shader_id;
    prg->num_inputs = ccf.num_inputs;
    prg->used_textures[0] = ccf.used_textures[0];
    prg->used_textures[1] = ccf.used_textures[1];
    prg->num_floats = num_floats;
    prg->library = library;
    prg->vertex_descriptor = vertexDescriptor;
    prg->source = shader_combined_source;

	auto vertexFunc = library->newFunction( NS::String::string("vertex_main", NS::UTF8StringEncoding));
	auto fragmentFunc = library->newFunction( NS::String::string("fragment_main", NS::UTF8StringEncoding));


	// create pipeline
	auto pipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
	pipelineDesc->setVertexFunction(vertexFunc);
	pipelineDesc->setFragmentFunction(fragmentFunc);
	pipelineDesc->setVertexDescriptor(prg->vertex_descriptor); 
	pipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

	auto pipelineState = mtl_state.device->newRenderPipelineState(pipelineDesc, &error);

	if (!pipelineState) {
		printf("failed to make pipeline. %s\n", error->localizedDescription()->utf8String());
		std::cout << prg->source;
		exit(-1);
	}

	prg->pipeline = pipelineState;

	printf("new shader made\n");
    gfx_metal_load_shader(reinterpret_cast<ShaderProgram *>(prg));
    return reinterpret_cast<ShaderProgram *>(prg);
}

static struct ShaderProgram *gfx_metal_lookup_shader(uint32_t shader_id) {
    for (size_t i = 0; i < mtl_state.shader_program_pool_size; i++) {
        if (mtl_state.shader_program_pool[i].shader_id == shader_id) {
            return reinterpret_cast<ShaderProgram *>(&mtl_state.shader_program_pool[i]);
        }
    }
    return nullptr;
}

static void gfx_metal_shader_get_info(ShaderProgram *prg, uint8_t *num_inputs, bool used_textures[2]) {
    ShaderProgramMetal *mtl_prg = reinterpret_cast<ShaderProgramMetal *>(prg);

    *num_inputs = mtl_prg->num_inputs;
    used_textures[0] = mtl_prg->used_textures[0];
    used_textures[1] = mtl_prg->used_textures[1];
}

static uint32_t gfx_metal_new_texture(void) {
    mtl_state.textures.resize(mtl_state.textures.size() + 1);
    return (uint32_t)(mtl_state.textures.size() - 1);
}

static void gfx_metal_select_texture(int tile, uint32_t texture_id) {
    mtl_state.current_tile = tile;
    mtl_state.current_texture_ids[tile] = texture_id;
}

static void gfx_metal_upload_texture(const uint8_t *rgba32_buf,
                                     int width,
                                     int height)
{
	printf("uploading texture\n");
    uint32_t texture_id =
        mtl_state.current_texture_ids[mtl_state.current_tile];

    TextureDataMetal &td = mtl_state.textures[texture_id];

    // Release old texture if replacing
    if (td.texture) {
        td.texture->release();
        td.texture = nullptr;
    }

    // Describe texture
    MTL::TextureDescriptor *desc =
        MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatRGBA8Unorm,
            width,
            height,
            false
        );

    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModeShared);

    td.texture = mtl_state.device->newTexture(desc);

    td.width = width;
    td.height = height;

    // Upload pixel data
    MTL::Region region = MTL::Region::Make2D(0, 0, width, height);

    td.texture->replaceRegion(
        region,
        0,                      // mip level
        rgba32_buf,
        width * 4               // bytes per row
    );

    desc->release();
}

static void gfx_metal_set_sampler_parameters(int sampler,
                                             bool linear_filter,
                                             uint32_t cms,
                                             uint32_t cmt) {
    (void)sampler;
    (void)linear_filter;
    (void)cms;
    (void)cmt;
    log_event("gfx_metal_set_sampler_parameters\n");
}

static void gfx_metal_set_depth_test(bool depth_test) {
    (void)depth_test;
    log_event("gfx_metal_set_depth_test\n");
}

static void gfx_metal_set_depth_mask(bool z_upd) {
    (void)z_upd;
    log_event("gfx_metal_set_depth_mask\n");
}

static void gfx_metal_set_zmode_decal(bool zmode_decal) {
    (void)zmode_decal;
    log_event("gfx_metal_set_zmode_decal\n");
}

static void gfx_metal_set_viewport(int x, int y, int width, int height) {
    (void)x;
    (void)y;
    (void)width;
    (void)height;

    MTL::Viewport viewport{
        static_cast<double>(x), 
        static_cast<double>(y), 
        static_cast<double>(width), 
        static_cast<double>(height)
    };
    mtl_state.viewport = viewport;
}

static void gfx_metal_set_scissor(int x, int y, int width, int height) {
    (void)x;
    (void)y;
    (void)width;
    (void)height;
    log_event("gfx_metal_set_scissor\n");
}

static void gfx_metal_set_use_alpha(bool use_alpha) {
    (void)use_alpha;
    log_event("gfx_metal_set_use_alpha\n");
}

static void gfx_metal_draw_triangles(float buf_vbo[],
                                     size_t buf_vbo_len,
                                     size_t buf_vbo_num_tris)
{
    if (!mtl_state.current_encoder) return;

    mtl_state.current_encoder->setRenderPipelineState(
        mtl_state.active_shader->pipeline);

	size_t size = buf_vbo_len * sizeof(float);

	// Align to 256 bytes (required by Metal)
	size_t aligned = (size + 255) & ~255;

	void* dst = (uint8_t*)mtl_state.dynamic_vertex_buffer->contents()
				+ mtl_state.dynamic_offset;

	memcpy(dst, buf_vbo, size);

	mtl_state.current_encoder->setVertexBuffer(
		mtl_state.dynamic_vertex_buffer,
		mtl_state.dynamic_offset,
		0
	);

    mtl_state.current_encoder->drawPrimitives(
        MTL::PrimitiveTypeTriangle,
		static_cast<NS::UInteger>(0),
        buf_vbo_num_tris * 3
    );

	mtl_state.dynamic_offset += aligned;

	uint32_t tex_id = mtl_state.current_texture_ids[0];

	if (tex_id < mtl_state.textures.size()) {
		TextureDataMetal &td = mtl_state.textures[tex_id];
		if (td.texture) {
			mtl_state.current_encoder->setFragmentTexture(td.texture, 0);
		}
	}
}


static void gfx_metal_init(void) {
    mtl_state.layer = static_cast<CA::MetalLayer *>(gfx_sdl_get_layer());
    mtl_state.device = MTL::CreateSystemDefaultDevice();
    mtl_state.layer->setDevice(mtl_state.device);
    mtl_state.queue = mtl_state.device->newCommandQueue();

	mtl_state.dynamic_vertex_buffer =
		mtl_state.device->newBuffer(
			1 * 1024 * 1024, // 1MB, adjust as needed
			MTL::ResourceStorageModeShared
		);

	mtl_state.dynamic_offset = 0;
}

static void gfx_metal_on_resize(void) {
}

static void gfx_metal_start_frame(void) {
    mtl_state.current_drawable = mtl_state.layer->nextDrawable();
    if (!mtl_state.current_drawable) return;

    mtl_state.current_pass_desc = MTL::RenderPassDescriptor::renderPassDescriptor();

    auto color = mtl_state.current_pass_desc->colorAttachments()->object(0);
    color->setTexture(mtl_state.current_drawable->texture());
    color->setLoadAction(MTL::LoadActionClear);
    color->setStoreAction(MTL::StoreActionStore);
    color->setClearColor(MTL::ClearColor(0.1, 0.1, 0.1, 1.0));

    mtl_state.current_cmd_buffer = mtl_state.queue->commandBuffer();

    mtl_state.current_encoder =
        mtl_state.current_cmd_buffer->renderCommandEncoder(
            mtl_state.current_pass_desc);

    mtl_state.current_encoder->setViewport(mtl_state.viewport);
}


static void gfx_metal_finish_render(void) {
	if (mtl_state.current_cmd_buffer) {
        mtl_state.current_cmd_buffer->waitUntilCompleted();
    }
}
static void gfx_metal_end_frame(void) {
    if (!mtl_state.current_encoder) return;

    mtl_state.current_encoder->endEncoding();

    mtl_state.current_cmd_buffer->presentDrawable(
        mtl_state.current_drawable);

    mtl_state.current_cmd_buffer->commit();

    // Reset per-frame state
    mtl_state.current_encoder = nullptr;
    mtl_state.current_cmd_buffer = nullptr;
    mtl_state.current_drawable = nullptr;
    mtl_state.current_pass_desc = nullptr;

	mtl_state.dynamic_offset = 0;
}

static void gfx_metal_shutdown(void) {
}

GfxRenderingAPI gfx_metal_api = {
    gfx_metal_z_is_from_0_to_1,
    gfx_metal_unload_shader,
    gfx_metal_load_shader,
    gfx_metal_create_and_load_new_shader,
    gfx_metal_lookup_shader,
    gfx_metal_shader_get_info,
    gfx_metal_new_texture,
    gfx_metal_select_texture,
    gfx_metal_upload_texture,
    gfx_metal_set_sampler_parameters,
    gfx_metal_set_depth_test,
    gfx_metal_set_depth_mask,
    gfx_metal_set_zmode_decal,
    gfx_metal_set_viewport,
    gfx_metal_set_scissor,
    gfx_metal_set_use_alpha,
    gfx_metal_draw_triangles,
    gfx_metal_init,
    gfx_metal_on_resize,
    gfx_metal_start_frame,
    gfx_metal_end_frame,
    gfx_metal_finish_render,
    gfx_metal_shutdown
};
#endif // RAPI_METAL