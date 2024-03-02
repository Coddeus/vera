//! This crate is where Vulkan is set up (`Vera::create()`) and core actions are handled (`show()`, `save()`, etc.)

// pub mod elements;
// pub use elements::*;

/// Buffers which will be sent to the GPU, and rely on the vulkano crate to be compiled
pub mod buffers;
#[allow(unused_imports)]
pub use buffers::*;
/// Matrices, which interpret the input transformations from vera, on which transformations are applied, and which are sent in buffers.
pub mod matrix;
pub use matrix::*;
/// "Transformers", update buffers of matrices and colors according to vera input transformations and colorizations: start/end time, speed evolution
pub mod transformer;
#[allow(unused_imports)]
pub use transformer::*;
use winit::monitor::{MonitorHandle, VideoMode};

use std::sync::Arc;
use std::time::Instant;

use vera::{Input, Model, Tf, Transformation, Evolution, Colorization, Cl};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::padded::Padded;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::depth_stencil::{DepthStencilState, DepthState};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer, BufferContents};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageFormatInfo, ImageType, ImageUsage, SampleCount};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    StandardMemoryAllocator, MemoryAllocator,
};
use vulkano::format::Format;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo, ComputePipeline,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use vulkano::swapchain::{
    acquire_next_image, PresentFuture, Surface, Swapchain, SwapchainAcquireFuture,
    SwapchainCreateInfo, SwapchainPresentInfo, PresentMode,
};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::{self, GpuFuture};
use vulkano::{DeviceSize, Validated, Version};

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Fullscreen, Window, WindowBuilder};

/// The struct the user interacts with.
pub struct Vera {
    event_loop: EventLoop<()>,
    vk: Vk,
}

/// The underlying base with executes specific tasks.
struct Vk {
    // ----- Created on initialization
    _library: Arc<vulkano::VulkanLibrary>,
    _instance: Arc<Instance>,
    _surface: Arc<Surface>,
    window: Arc<Window>,

    _physical_device: Arc<PhysicalDevice>,
    _queue_family_index: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,

    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    cb_allocator: StandardCommandBufferAllocator,
    ds_allocator: StandardDescriptorSetAllocator,


    recreate_swapchain: bool,
    show_count: u32,
    time: f32,


    draw_pipeline: Arc<GraphicsPipeline>,
    draw_descriptor_set: Arc<PersistentDescriptorSet>,
    vertex_compute_pipeline: Arc<ComputePipeline>,
    vertex_compute_descriptor_set: Arc<PersistentDescriptorSet>,
    model_compute_pipeline: Arc<ComputePipeline>,
    model_compute_descriptor_set: Arc<PersistentDescriptorSet>,

    drawing_vs: EntryPoint,
    drawing_fs: EntryPoint,
    drawing_fences: Vec<
        Option<
            FenceSignalFuture<
                PresentFuture<
                    CommandBufferExecFuture<
                        JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>,
                    >,
                >
            >,
        >,
    >,
    previous_drawing_fence_i: u32,
    
    // -----
    background_color: [f32; 4],
    start_time: f32,
    end_time: f32,
    vertex_dispatch_len: u32,
    model_dispatch_len: u32,



    general_push_cs: CSGeneral,
    general_push_vs: VSGeneral,
    general_push_transformer: (Transformer, Transformer),


    vsinput_buffer: Subbuffer<[BaseVertex]>,
    basevertex_buffer: Subbuffer<[BaseVertex]>,
    entity_buffer: Subbuffer<[Entity]>,
    modelt_buffer: Subbuffer<[MatrixT]>,

    vertex_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]>,
    vertex_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]>,
    vertex_colortransformation_buffer: Subbuffer<[ColorTransformation]>,
    vertex_colortransformer_buffer: Subbuffer<[ColorTransformer]>,

    model_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]>,
    model_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]>,


    text_texture: Arc<ImageView>,
    text_sampler: Arc<Sampler>,
}

mod vertex_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460
            #define PI 3.1415926535

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


            // STRUCTS

            struct BaseVertex {
                vec4 position;
                vec4 color;
                vec2 tex_coord; // Index in t.d
                uint tex_id;
                uint entity_id;
            };
            struct MatrixTransformation {
                vec3 val; // The values of this transformation, interpreted depending on the type.
                uint ty; // The type of matrix transformation
                float start;
                float end;
                uint evolution;
            };
            struct MatrixTransformer {
                mat4 current; // The current resulting matrix, acts a pre-result cache.
                uvec2 range; // The index range of the matrix transformations of this transformer inside `t`.
            };
            struct ColorTransformation {
                vec4 val; // The values of this transformation, interpreted depending on the type.
                uint ty; // The type of color transformation
                float start;
                float end;
                uint evolution;
            };
            struct ColorTransformer {
                vec4 current; // The current resulting color, acts a pre-output cache.
                uvec2 range; // The index range of the color transformations of this transformer inside `c`.
            };


            // BUFFERS

            // Per-vertex
            layout(set = 0, binding = 0) buffer OutputVertices {
                BaseVertex d[]; // 'd' for 'Data'.
            } ov;
            layout(set = 0, binding = 1) readonly buffer InputVertices {
                BaseVertex d[];
            } iv;
            layout(set = 0, binding = 2) buffer MatrixTransformers {
                MatrixTransformer d[];
            } tf;
            layout(set = 0, binding = 3) readonly buffer MatrixTransformations {
                MatrixTransformation d[];
            } t;
            layout(set = 0, binding = 4) buffer ColorTransformers {
                ColorTransformer d[];
            } cl;
            layout(set = 0, binding = 5) readonly buffer ColorTransformations {
                ColorTransformation d[];
            } c;
            // Unique
            layout( push_constant ) uniform GeneralInfo {
                float time;
            } gen;


            // TRANSFORMATIONS

            vec4 interpolate(vec4 start_color, vec4 end_color, float advancement) {
                return vec4(
                    start_color[0] * (1.0-advancement) + end_color[0] * advancement,
                    start_color[1] * (1.0-advancement) + end_color[1] * advancement,
                    start_color[2] * (1.0-advancement) + end_color[2] * advancement,
                    start_color[3] * (1.0-advancement) + end_color[3] * advancement
                );
                // return mix(start_color, end_color, advancement);
            }

            mat4 scale(float x_scale, float y_scale, float z_scale) {
                return mat4(
                    x_scale ,   0.0     ,   0.0     , 0.0 , 
                    0.0     ,   y_scale ,   0.0     , 0.0 , 
                    0.0     ,   0.0     ,   z_scale , 0.0 , 
                    0.0     ,   0.0     ,   0.0     , 1.0
                );
            }

            mat4 translate(float x_move, float y_move, float z_move) {
                return mat4(
                    1.0 ,   0.0 ,   0.0 ,   x_move ,
                    0.0 ,   1.0 ,   0.0 ,   y_move ,
                    0.0 ,   0.0 ,   1.0 ,   z_move ,
                    0.0 ,   0.0 ,   0.0 ,   1.0
                );
            }

            mat4 rotate_x(float angle) {
                return mat4(
                    1.0     ,   0.0         ,   0.0         , 0.0 , 
                    0.0     ,   cos(angle)  ,   sin(angle)  , 0.0 , 
                    0.0     ,   -sin(angle) ,   cos(angle)  , 0.0 , 
                    0.0     ,   0.0         ,   0.0         , 1.0
                );
            }

            mat4 rotate_y(float angle) {
                return mat4(
                    cos(angle)  ,   0.0 ,   sin(angle)  , 0.0 , 
                    0.0         ,   1.0 ,   0.0         , 0.0 , 
                    -sin(angle) ,   0.0 ,   cos(angle)  , 0.0 , 
                    0.0         ,   0.0 ,   0.0         , 1.0
                );
            }

            mat4 rotate_z(float angle) {
                return mat4(
                    cos(angle)  ,   sin(angle)  ,   0.0 ,   0.0 , 
                    -sin(angle) ,   cos(angle)  ,   0.0 ,   0.0 , 
                    0.0         ,   0.0         ,   1.0 ,   0.0 , 
                    0.0         ,   0.0         ,   0.0 ,   1.0
                );
            }


            // EVOLUTIONS

            float advancement(float start, float end, uint e) {
                if (gen.time >= end) {
                    return 1.0;
                }
                float init = (gen.time-start)/(end-start);

                if (e==0) {
                    return init;
                } else if (e==1) {
                    return sin(init * PI / 2.0);
                } else if (e==2) {
                    return 1.0 - cos(init * PI / 2.0);
                } else if (e==3) {
                    return (sin((init - 0.5) * PI) + 1.0) / 2.0;
                } else if (e==4) {
                    if (init < 0.5) { return sin(init * PI) / 2.0; }
                    else { return 0.5 + (1.0 - sin(init * PI)) / 2.0; }
                } else {
                    return init;
                }
            }

            // TRANSFORMATIONS MATCHING

            mat4 vertex_matrix_transformation(uint i) {
                float adv = advancement(t.d[i].start, t.d[i].end, t.d[i].evolution);
                if (t.d[i].ty==0) {
                    return scale(t.d[i].val.x * adv + 1.0 * (1.0-adv), t.d[i].val.y * adv + 1.0 * (1.0-adv), t.d[i].val.z * adv + 1.0 * (1.0-adv));
                } else if (t.d[i].ty==1) {
                    return translate(t.d[i].val.x * adv, t.d[i].val.y * adv, t.d[i].val.z * adv);
                } else if (t.d[i].ty==2) {
                    return rotate_x(t.d[i].val.x * adv);
                } else if (t.d[i].ty==3) {
                    return rotate_y(t.d[i].val.x * adv);
                } else {
                    return rotate_z(t.d[i].val.x * adv);
                }
            }

            vec4 vertex_color_transformation(uint i, vec4 out_color) {
                float adv = advancement(c.d[i].start, c.d[i].end, c.d[i].evolution);
                if (c.d[i].ty==0) {
                    return interpolate(out_color, c.d[i].val, adv);
                }
            }


            // TRANSFORMING

            void transform_vertex() {
                // position
                bool first = true;
                vec4 out_position = iv.d[gl_GlobalInvocationID.x].position * tf.d[gl_GlobalInvocationID.x].current;
                for (uint i=tf.d[gl_GlobalInvocationID.x].range.x; i<tf.d[gl_GlobalInvocationID.x].range.y; i++) {
                    if (t.d[i].start<gen.time) {
                        mat4 mat = vertex_matrix_transformation(i);
                        // if (first && t.d[i].end<gen.time) {
                        //     tf.d[gl_GlobalInvocationID.x].current *= mat;
                        //     tf.d[gl_GlobalInvocationID.x].range.x += 1;
                        // } else {
                        //     first = false;
                        // }
                        out_position *= mat;
                    }
                }
                ov.d[gl_GlobalInvocationID.x].position = out_position;
                
                // color
                first = true;
                vec4 out_color = cl.d[gl_GlobalInvocationID.x].current;
                for (uint i=cl.d[gl_GlobalInvocationID.x].range.x; i<cl.d[gl_GlobalInvocationID.x].range.y; i++) {
                    if (c.d[i].start<gen.time) {
                        vec4 vec = vertex_color_transformation(i, out_color);
                        // if (first && c.d[i].end<gen.time) {
                        //     cl.d[gl_GlobalInvocationID.x].current = vec;
                        //     cl.d[gl_GlobalInvocationID.x].range.x += 1;
                        // } else {
                        //     first = false;
                        // }
                        out_color = vec;
                    }
                }
                ov.d[gl_GlobalInvocationID.x].color = out_color;
            }


            // MAIN

            void main() {
                // Calculate & fill vertex position and color.
                transform_vertex();
            }
        ",
    }
}

mod model_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460
            #define PI 3.1415926535

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


            // STRUCTS

            struct ModelT {
                mat4 mat;
            };
            struct MatrixTransformation {
                vec3 val; // The values of this transformation, interpreted depending on the type.
                uint ty; // The type of matrix transformation
                float start;
                float end;
                uint evolution;
            };
            struct MatrixTransformer {
                mat4 current; // The current resulting matrix, acts a pre-result cache.
                uvec2 range; // The index range of the matrix transformations of this transformer inside `t`.
            };


            // BUFFERS

            // Per-model
            layout(set = 0, binding = 7) buffer OutputModels {
                ModelT d[];
            } om;
            layout(set = 0, binding = 8) buffer Model_MatrixTransformers {
                MatrixTransformer d[];
            } m_tf;
            layout(set = 0, binding = 9) readonly buffer Model_MatrixTransformations {
                MatrixTransformation d[];
            } m_t;
            layout( push_constant ) uniform GeneralInfo {
                float time;
            } gen;


            // TRANSFORMATIONS

            mat4 scale(float x_scale, float y_scale, float z_scale) {
                return mat4(
                    x_scale ,   0.0     ,   0.0     , 0.0 , 
                    0.0     ,   y_scale ,   0.0     , 0.0 , 
                    0.0     ,   0.0     ,   z_scale , 0.0 , 
                    0.0     ,   0.0     ,   0.0     , 1.0
                );
            }

            mat4 translate(float x_move, float y_move, float z_move) {
                return mat4(
                    1.0 ,   0.0 ,   0.0 ,   x_move ,
                    0.0 ,   1.0 ,   0.0 ,   y_move ,
                    0.0 ,   0.0 ,   1.0 ,   z_move ,
                    0.0 ,   0.0 ,   0.0 ,   1.0
                );
            }

            mat4 rotate_x(float angle) {
                return mat4(
                    1.0     ,   0.0         ,   0.0         , 0.0 , 
                    0.0     ,   cos(angle)  ,   sin(angle)  , 0.0 , 
                    0.0     ,   -sin(angle) ,   cos(angle)  , 0.0 , 
                    0.0     ,   0.0         ,   0.0         , 1.0
                );
            }

            mat4 rotate_y(float angle) {
                return mat4(
                    cos(angle)  ,   0.0 ,   sin(angle)  , 0.0 , 
                    0.0         ,   1.0 ,   0.0         , 0.0 , 
                    -sin(angle) ,   0.0 ,   cos(angle)  , 0.0 , 
                    0.0         ,   0.0 ,   0.0         , 1.0
                );
            }

            mat4 rotate_z(float angle) {
                return mat4(
                    cos(angle)  ,   sin(angle)  ,   0.0 ,   0.0 , 
                    -sin(angle) ,   cos(angle)  ,   0.0 ,   0.0 , 
                    0.0         ,   0.0         ,   1.0 ,   0.0 , 
                    0.0         ,   0.0         ,   0.0 ,   1.0
                );
            }


            // EVOLUTIONS

            float advancement(float start, float end, uint e) {
                if (gen.time >= end) {
                    return 1.0;
                }
                float init = (gen.time-start)/(end-start);

                if (e==0) {
                    return init;
                } else if (e==1) {
                    return sin(init * PI / 2.0);
                } else if (e==2) {
                    return 1.0 - cos(init * PI / 2.0);
                } else if (e==3) {
                    return (sin((init - 0.5) * PI) + 1.0) / 2.0;
                } else if (e==4) {
                    if (init < 0.5) { return sin(init * PI) / 2.0; }
                    else { return 0.5 + (1.0 - sin(init * PI)) / 2.0; }
                } else {
                    return init;
                }
            }

            // TRANSFORMATIONS MATCHING

            mat4 model_matrix_transformation(uint i) {
                float adv = advancement(m_t.d[i].start, m_t.d[i].end, m_t.d[i].evolution);
                if (m_t.d[i].ty==0) {
                    return scale(m_t.d[i].val.x * adv + 1.0 * (1.0-adv), m_t.d[i].val.y * adv + 1.0 * (1.0-adv), m_t.d[i].val.z * adv + 1.0 * (1.0-adv));
                } else if (m_t.d[i].ty==1) {
                    return translate(m_t.d[i].val.x * adv, m_t.d[i].val.y * adv, m_t.d[i].val.z * adv);
                } else if (m_t.d[i].ty==2) {
                    return rotate_x(m_t.d[i].val.x * adv);
                } else if (m_t.d[i].ty==3) {
                    return rotate_y(m_t.d[i].val.x * adv);
                } else {
                    return rotate_z(m_t.d[i].val.x * adv);
                }
            }


            // TRANSFORMING

            void transform_model() {
                bool first = true;
                uint entity_id = gl_GlobalInvocationID.x;
                mat4 out_model = m_tf.d[entity_id].current;
                for (uint i=m_tf.d[entity_id].range.x; i<m_tf.d[entity_id].range.y; i++) {
                    if (m_t.d[i].start<gen.time) {
                        mat4 mat = model_matrix_transformation(i);
                        // if (first && m_t.d[i].end<gen.time) {
                        //     m_tf.d[entity_id].current *= mat;
                        //     m_tf.d[entity_id].range.x += 1;
                        // } else {
                        //     first = false;
                        // }
                        out_model *= mat;
                    }
                }
                om.d[entity_id].mat = out_model;
            }


            // MAIN

            void main() {
                // Calculate & fill model matrix
                transform_model();
            }
        ",
    }
}

mod vs { 
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            struct Entity {
                uvec4 parent_id;
            };
            struct ModelT {
                mat4 mat;
            };

            //  Inputs
            // Vertex
            layout(location = 0) in vec4 position;
            layout(location = 1) in vec4 color;
            // Texture
            layout(location = 4) in vec2 tex_coord;
            layout(location = 3) in uint tex_id;
            // Model
            layout(location = 2) in uint entity_id;

            //  Outputs
            layout(location = 0) out vec4 out_color;
            layout(location = 1) out uint tex_id_out;
            layout(location = 2) out vec2 tex_coord_out;

            layout(set = 0, binding = 6) readonly buffer Entities {
                Entity data[];
            } ent;
            layout(set = 0, binding = 7) readonly buffer ModelTransformations {
                ModelT data[];
            } _mod;
            layout(push_constant) uniform GeneralInfo {
                mat4 mat;
            } gen;

            void main() {
                vec4 pos = position;
                out_color = color;
                uint model_id=entity_id;

                while (model_id>0) {
                    pos *= _mod.data[model_id].mat;
                    model_id=ent.data[model_id].parent_id.x;
                }

                pos *= gen.mat;
                
                gl_Position = pos;


                tex_id_out = tex_id;
                tex_coord_out = tex_coord;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            layout(location = 0) in vec4 in_color;
            layout(location = 1) in flat uint tex_id;
            layout(location = 2) in vec2 tex_coords;

            layout(set = 0, binding = 10) uniform sampler s;
            layout(set = 0, binding = 11) uniform texture2D tex;

            const vec2 unitRange = vec2(16.0/1920.0, 16.0/1920.0);


            float median(float r, float g, float b) {
                return max(min(r, g), min(max(r, g), b));
            }

            float screenPxRange() {
                vec2 screenTexSize = vec2(1.0)/fwidth(tex_coords);
                return max(0.5*dot(unitRange, screenTexSize), 1.0);
            }

            void main() {
                f_color = in_color;
                if (tex_id == 1) { // Is text
                    // Signed distance field
                    vec4 mtsd = texture(sampler2D(tex, s), tex_coords);

                    // float rgb_sd = median(mtsd.r, mtsd.g, mtsd.b);
                    // float screenPxDistance = screenPxRange()*(rgb_sd - 0.5);

                    float alpha_sd = mtsd.a;
                    float screenPxDistanceAlpha = screenPxRange()*(alpha_sd - 0.5);

                    float opacity = clamp(screenPxDistanceAlpha, 0.0, 1.0);
                    f_color = vec4(in_color.rgb, opacity * in_color.a);
                    
                    // ----------------- Not working
                    // // Simple signed distance field
                    // float sd = texture(sampler2D(tex, s), tex_coords).r;
                    // float screenPxDistance = screenPxRange() * sd;
                    // float opacity = clamp(screenPxDistance, 0.0, 1.0);
                    // f_color = vec4(in_color.rgb, in_color.a * opacity);


                    // // Other ?  - Multi-channel

                    // // float rgb_sd = median(mtsd.r, mtsd.g, mtsd.b);
                    // // float screenPxDistance = screenPxRange()*(rgb_sd - 0.5);
                }
            }
        ",
    }
}

impl Vera {
    /// Sets up Vera with Vulkan
    /// - `input` defines what's to be drawn and how.
    pub fn init(input: Input) -> Self {
        // Event_loop/extensions/instance/surface/window/physical_device/queue_family/device/queue/swapchain/images/render_pass/framebuffers
        // ---------------------------------------------------------------------------------------------------------------------------------
        let event_loop = EventLoop::new();
        let _library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(&event_loop);
        let _instance = Instance::new(
            _library.clone(),
            InstanceCreateInfo {
                application_name: Some("Vera".to_owned()),
                application_version: Version::major_minor(0, 2),
                enabled_extensions: required_extensions,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(LogicalSize {
                    width: 1900,
                    height: 1000,
                })
                .with_resizable(true)
                .with_decorations(false)         // TODOFEATURES
                .with_title("Vera")
                .with_transparent(false)
                .with_maximized(true)
                .build(&event_loop)
                .unwrap(),
        );
        let _surface = Surface::from_window(_instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (_physical_device, _queue_family_index) = _instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && q.queue_flags.contains(QueueFlags::COMPUTE)
                            && q.queue_flags.contains(QueueFlags::TRANSFER)
                            && p.surface_support(i as u32, &_surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available");

        println!(
            "Using device: {} (type: {:?})",
            _physical_device.properties().device_name,
            _physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            _physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: _queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        // Allocators
        // ----------
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let ds_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        // --------

        let caps = _physical_device
            .surface_capabilities(&_surface, Default::default())
            .expect("failed to get surface capabilities");
        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = _physical_device
            .clone()
            .surface_formats(&_surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            _surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count.max(2),
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::STORAGE,
                composite_alpha,
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
        )
        .unwrap();

        let extent: [u32; 3] = images[0].extent();

        let msaa_image: Arc<ImageView> = ImageView::new_default(
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: swapchain.image_format(),
                    extent,
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    samples: SampleCount::Sample8,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let msaa_depth_attachment = ImageView::new_default(
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D16_UNORM,
                    extent,
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    samples: SampleCount::Sample8,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                msaa_color: {
                    format: swapchain.image_format(),
                    samples: 8,
                    load_op: Clear,
                    store_op: DontCare,
                },
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
                msaa_depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 8,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [msaa_color],
                color_resolve: [color],
                depth_stencil: {msaa_depth_stencil},
            },
        )
        .unwrap();

        let framebuffers: Vec<Arc<Framebuffer>> = images
            .iter()
            .map(|image| {
                let final_view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![msaa_image.clone(), final_view, msaa_depth_attachment.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<Arc<Framebuffer>>>();

        // ---------------------------------------------------------------------------------------------------------------------------------

        // Buffers
        // -------
        let (
            (background_color, start_time, end_time, vertex_dispatch_len, model_dispatch_len),
            (general_push_cs, general_push_vs, general_push_transformer),
            (
                vsinput_buffer,
                basevertex_buffer,
                entity_buffer,
                modelt_buffer,
    
                vertex_matrixtransformation_buffer,
                vertex_matrixtransformer_buffer,
                vertex_colortransformation_buffer,
                vertex_colortransformer_buffer,
    
                model_matrixtransformation_buffer,
                model_matrixtransformer_buffer,
            ),
        ) = from_input(queue.clone(), memory_allocator.clone(), &cb_allocator, input);

        let mut uploads = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let text_texture: Arc<ImageView> = {
            let png_bytes = include_bytes!("fonts/cmunti_mtsdf_128_16.png").as_slice();
            let decoder = png::Decoder::new(png_bytes);
            let mut reader = decoder.read_info().unwrap();
            let info = reader.info();
            // let channels = info.color_type;
            // println!("{:?}", channels);
            let extent = [info.width, info.height, 1];
    
            let upload_buffer = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (info.width * info.height * 4) as DeviceSize,
            )
            .unwrap();
    
            reader
                .next_frame(&mut upload_buffer.write().unwrap())
                .unwrap();
    
            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_SRGB,
                    extent,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();
    
            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    image.clone(),
                ))
                .unwrap();
            uploads
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
    
            ImageView::new_default(image).unwrap()
        };
    
        let text_sampler: Arc<Sampler> = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        // -------

        // Graphics pipeline & Drawing command buffer
        // ------------------------------------------
        let drawing_vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create vertex shader module");
        let drawing_fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create fragment shader module");
        let vertex_cs = vertex_cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create vertex compute shader module");
        let model_cs = model_cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create model compute shader module");


        let draw_pipeline: Arc<GraphicsPipeline> = {
            let vertex_input_state = BaseVertex::per_vertex()
                .definition(&drawing_vs.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(drawing_vs.clone()),
                PipelineShaderStageCreateInfo::new(drawing_fs.clone()),
            ];

            let pipeline_layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState {
                        rasterization_samples: subpass.num_samples().unwrap(),
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(subpass.into()),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
                },
            )
            .unwrap()
        };
        let draw_descriptor_set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
            &ds_allocator,
            draw_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                // WriteDescriptorSet::buffer(0, vsinput_buffer.clone()),
                // WriteDescriptorSet::buffer(1, basevertex_buffer.clone()),
                // WriteDescriptorSet::buffer(2, vertex_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(3, vertex_matrixtransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(4, vertex_colortransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(5, vertex_colortransformation_buffer.clone()),
                WriteDescriptorSet::buffer(6, entity_buffer.clone()),
                WriteDescriptorSet::buffer(7, modelt_buffer.clone()),
                // WriteDescriptorSet::buffer(8, model_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(9, model_matrixtransformation_buffer.clone()),
                WriteDescriptorSet::sampler(10, text_sampler.clone()),
                WriteDescriptorSet::image_view(11, text_texture.clone()),
            ],
            [],
        )
        .unwrap();


        let vertex_compute_pipeline: Arc<ComputePipeline> = {
            let stage = PipelineShaderStageCreateInfo::new(vertex_cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };
        let vertex_compute_descriptor_set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
            &ds_allocator,
            vertex_compute_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, vsinput_buffer.clone()),
                WriteDescriptorSet::buffer(1, basevertex_buffer.clone()),
                WriteDescriptorSet::buffer(2, vertex_matrixtransformer_buffer.clone()),
                WriteDescriptorSet::buffer(3, vertex_matrixtransformation_buffer.clone()),
                WriteDescriptorSet::buffer(4, vertex_colortransformer_buffer.clone()),
                WriteDescriptorSet::buffer(5, vertex_colortransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(6, entity_buffer.clone()),
                // WriteDescriptorSet::buffer(7, modelt_buffer.clone()),
                // WriteDescriptorSet::buffer(8, model_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(9, model_matrixtransformation_buffer.clone()),
            ],
            [],
        )
        .unwrap();


        let model_compute_pipeline: Arc<ComputePipeline> = {
            let stage = PipelineShaderStageCreateInfo::new(model_cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };
        let model_compute_descriptor_set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
            &ds_allocator,
            model_compute_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                // WriteDescriptorSet::buffer(0, vsinput_buffer.clone()),
                // WriteDescriptorSet::buffer(1, basevertex_buffer.clone()),
                // WriteDescriptorSet::buffer(2, vertex_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(3, vertex_matrixtransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(4, vertex_colortransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(5, vertex_colortransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(6, entity_buffer.clone()),
                WriteDescriptorSet::buffer(7, modelt_buffer.clone()),
                WriteDescriptorSet::buffer(8, model_matrixtransformer_buffer.clone()),
                WriteDescriptorSet::buffer(9, model_matrixtransformation_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        let drawing_fences: Vec<Option<FenceSignalFuture<_>>> = (0..framebuffers.len()).map(|_| None).collect();
        let previous_drawing_fence_i: u32 = 0;

        // ------------------------------------------

        // Window-related updates
        // ----------------------
        let recreate_swapchain: bool = false;
        let show_count: u32 = 0;
        let time: f32 = 0.0;

        // ----------------------

        Vera {
            event_loop,
            vk: Vk {
                _library,
                _instance,
                _surface,
                window,

                _physical_device,
                _queue_family_index,
                device,
                queue,

                swapchain,
                images,
                render_pass,
                framebuffers,


                memory_allocator,
                cb_allocator,
                ds_allocator,


                recreate_swapchain,
                show_count,
                time,


                draw_pipeline,
                draw_descriptor_set,
                vertex_compute_pipeline,
                vertex_compute_descriptor_set,
                model_compute_pipeline,
                model_compute_descriptor_set,


                drawing_vs,
                drawing_fs,
                drawing_fences,
                previous_drawing_fence_i,


                background_color,
                start_time,
                end_time,
                vertex_dispatch_len,
                model_dispatch_len,


                general_push_cs,
                general_push_vs,
                general_push_transformer,


                vsinput_buffer,
                basevertex_buffer,
                entity_buffer,
                modelt_buffer,

                vertex_matrixtransformation_buffer,
                vertex_matrixtransformer_buffer,
                vertex_colortransformation_buffer,
                vertex_colortransformer_buffer,

                model_matrixtransformation_buffer,
                model_matrixtransformer_buffer,


                text_texture,
                text_sampler,
            },
        }
    }

    /// Resets the animation data from `input`, which is consumed.
    /// 
    /// In heavy animations, resetting is useful for reducing the CPU/GPU usage if you split you animations in several parts.
    /// Also useful if called from a loop for hot-reloading.
    pub fn reset(&mut self, input: Input) {
        // Clean previous fences data
        for fence_index in 0..self.vk.drawing_fences.len() {
            if let Some(image_fence) = &mut self.vk.drawing_fences[fence_index as usize] {
                image_fence.wait(None).unwrap();
                image_fence.cleanup_finished();
            }
        }

        self.vk.recreate_swapchain();


        // Reset buffers data
        (
            (self.vk.background_color, self.vk.start_time, self.vk.end_time, self.vk.vertex_dispatch_len, self.vk.model_dispatch_len),
            (self.vk.general_push_cs, self.vk.general_push_vs, self.vk.general_push_transformer),
            (
                self.vk.vsinput_buffer,
                self.vk.basevertex_buffer,
                self.vk.entity_buffer,
                self.vk.modelt_buffer,
    
                self.vk.vertex_matrixtransformation_buffer,
                self.vk.vertex_matrixtransformer_buffer,
                self.vk.vertex_colortransformation_buffer,
                self.vk.vertex_colortransformer_buffer,
    
                self.vk.model_matrixtransformation_buffer,
                self.vk.model_matrixtransformer_buffer,
            ),
        ) = from_input(self.vk.queue.clone(), self.vk.memory_allocator.clone(), &self.vk.cb_allocator, input);


        // Update descriptor sets to point to the new buffers.
        self.vk.draw_descriptor_set = PersistentDescriptorSet::new(
            &self.vk.ds_allocator,
            self.vk.draw_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                // WriteDescriptorSet::buffer(0, self.vk.vsinput_buffer.clone()),
                // WriteDescriptorSet::buffer(1, self.vk.basevertex_buffer.clone()),
                // WriteDescriptorSet::buffer(2, self.vk.vertex_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(3, self.vk.vertex_matrixtransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(4, self.vk.vertex_colortransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(5, self.vk.vertex_colortransformation_buffer.clone()),
                WriteDescriptorSet::buffer(6, self.vk.entity_buffer.clone()),
                WriteDescriptorSet::buffer(7, self.vk.modelt_buffer.clone()),
                // WriteDescriptorSet::buffer(8, self.vk.model_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(9, self.vk.model_matrixtransformation_buffer.clone()),
                WriteDescriptorSet::sampler(10, self.vk.text_sampler.clone()),
                WriteDescriptorSet::image_view(11, self.vk.text_texture.clone()),
            ],
            [],
        )
        .unwrap();
        self.vk.vertex_compute_descriptor_set = PersistentDescriptorSet::new(
            &self.vk.ds_allocator,
            self.vk.vertex_compute_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, self.vk.vsinput_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.vk.basevertex_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.vk.vertex_matrixtransformer_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.vk.vertex_matrixtransformation_buffer.clone()),
                WriteDescriptorSet::buffer(4, self.vk.vertex_colortransformer_buffer.clone()),
                WriteDescriptorSet::buffer(5, self.vk.vertex_colortransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(6, self.vk.entity_buffer.clone()),
                // WriteDescriptorSet::buffer(7, self.vk.modelt_buffer.clone()),
                // WriteDescriptorSet::buffer(8, self.vk.model_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(9, self.vk.model_matrixtransformation_buffer.clone()),
            ],
            [],
        )
        .unwrap();
        self.vk.model_compute_descriptor_set = PersistentDescriptorSet::new(
            &self.vk.ds_allocator,
            self.vk.model_compute_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                // WriteDescriptorSet::buffer(0, self.vk.vsinput_buffer.clone()),
                // WriteDescriptorSet::buffer(1, self.vk.basevertex_buffer.clone()),
                // WriteDescriptorSet::buffer(2, self.vk.vertex_matrixtransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(3, self.vk.vertex_matrixtransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(4, self.vk.vertex_colortransformer_buffer.clone()),
                // WriteDescriptorSet::buffer(5, self.vk.vertex_colortransformation_buffer.clone()),
                // WriteDescriptorSet::buffer(6, self.vk.entity_buffer.clone()),
                WriteDescriptorSet::buffer(7, self.vk.modelt_buffer.clone()),
                WriteDescriptorSet::buffer(8, self.vk.model_matrixtransformer_buffer.clone()),
                WriteDescriptorSet::buffer(9, self.vk.model_matrixtransformation_buffer.clone()),
            ],
            [],
        )
        .unwrap();
    }

    /// Shows the animation form start to end once.
    /// - Panics if there was an unexpected exit code from the event loop
    /// - Returns `false` if the window was closed while showing.
    /// - Returns true otherwise.
    pub fn show(&mut self) -> bool {
        match self.vk.show(&mut self.event_loop) {
            0 => {
                // Successfully finished.
                true
            }
            1 => {
                // Window closed
                println!("\nℹ Window closed.");
                false
            }
            n => {
                println!("🛑 Unexpected return code \"{}\" when running the main loop", n);
                false
            }
        }
    }

    pub fn save(&mut self,/*  width: u32, height: u32 */) {
        match self.vk.show(&mut self.event_loop) { // , (width, height)
            0 => {
                // Successfully finished
                println!("✨ Saved video!");
            }
            1 => {
                // Window closed
                println!("⁉ Window closed. Stopping encoding now.");
            }
            _ => {
                panic!("🛑 Unexpected return code when running the main loop");
            }
        }
    }
}

/// Treats `input`, and returns created buffers and metadata.
fn from_input(
    queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    cb_allocator: &StandardCommandBufferAllocator,
    input: Input
) -> (
    ([f32; 4], f32, f32, u32, u32),
    (CSGeneral, VSGeneral, (Transformer, Transformer)),
    (
        Subbuffer<[BaseVertex]>,
        Subbuffer<[BaseVertex]>,
        Subbuffer<[Entity]>,
        Subbuffer<[MatrixT]>,

        Subbuffer<[MatrixTransformation]>,
        Subbuffer<[MatrixTransformer]>,
        Subbuffer<[ColorTransformation]>,
        Subbuffer<[ColorTransformer]>,

        Subbuffer<[MatrixTransformation]>,
        Subbuffer<[MatrixTransformer]>,
    ),
) {
    // Meta and general (CPU and push constants)
    let view_t = input.v.own_fields();
    let projection_t = input.p.own_fields();

    let general_push_vs: VSGeneral = VSGeneral {
        mat: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    };
    let general_push_cs: CSGeneral = CSGeneral {
        time: 0.0,
    };
    let general_push_transformer: (Transformer, Transformer) = (Transformer::from_t(view_t), Transformer::from_t(projection_t)); // (View, Projection)


    let mut entity_index: u32 = 0;
    let mut vertex_matrixtransformation_offset: u32 = 0;
    let mut vertex_colortransformation_offset: u32 = 0;
    let mut model_matrixtransformation_offset: u32 = 0;

    let (
        mut basevertex_data,
            entity_data,
        mut vertex_matrixtransformation_data,
        mut vertex_matrixtransformer_data,
        mut vertex_colortransformation_data,
        mut vertex_colortransformer_data,
        mut model_matrixtransformation_data,
        mut model_matrixtransformer_data
    ) = from_model(Model::from_models(input.m), 0, &mut entity_index, &mut vertex_matrixtransformation_offset, &mut vertex_colortransformation_offset, &mut model_matrixtransformation_offset);

    fn from_model(
        model: Model,
        parent_id: u32, 
        entity_index: &mut u32,
        vertex_matrixtransformation_offset: &mut u32,
        vertex_colortransformation_offset: &mut u32,
        model_matrixtransformation_offset: &mut u32,
    ) -> (Vec<BaseVertex>, Vec<Entity>, Vec<MatrixTransformation>, Vec<MatrixTransformer>, Vec<ColorTransformation>, Vec<ColorTransformer>, Vec<MatrixTransformation>, Vec<MatrixTransformer>) {
        let current_id= *entity_index;
        *entity_index+=1;

        let (
            m_models,
            m_vertices,
            m_t,
        ) = model.own_fields();

        let mut basevertex: Vec<BaseVertex> = vec![];
        let mut entity: Vec<Entity> = vec![];
    
        let mut vertex_matrixtransformation: Vec<MatrixTransformation> = vec![];
        let mut vertex_matrixtransformer: Vec<MatrixTransformer> = vec![];
        let mut vertex_colortransformation: Vec<ColorTransformation> = vec![];
        let mut vertex_colortransformer: Vec<ColorTransformer> = vec![];
    
        let mut model_matrixtransformation: Vec<MatrixTransformation> = to_gpu_tf(m_t);
        let mmt_len = model_matrixtransformation.len() as u32;
        let mut model_matrixtransformer: Vec<MatrixTransformer> = vec![MatrixTransformer::from_lo(mmt_len, *model_matrixtransformation_offset)];
        *model_matrixtransformation_offset+=mmt_len;

        for v in m_vertices.into_iter() {
            let (
                v_position,
                v_color,
                v_tex_coord,
                v_tex_id,
                v_t,
                v_c,
            ) = v.own_fields();
            basevertex.push(BaseVertex {
                position: v_position,
                color: v_color.clone(),
                tex_coord: v_tex_coord,
                tex_id: v_tex_id,
                entity_id: current_id,
            });

            let gpu_tf = to_gpu_tf(v_t);
            let vmt_len = gpu_tf.len() as u32;
            vertex_matrixtransformation.extend(gpu_tf);
            vertex_matrixtransformer.push(MatrixTransformer::from_lo(vmt_len, *vertex_matrixtransformation_offset));
            *vertex_matrixtransformation_offset+=vmt_len;
            
            let gpu_cl = to_gpu_cl(v_c);
            let vmt_len = gpu_cl.len() as u32;
            vertex_colortransformation.extend(gpu_cl);
            vertex_colortransformer.push(ColorTransformer::from_loc(vmt_len, *vertex_colortransformation_offset, v_color));
            *vertex_colortransformation_offset+=vmt_len;
        }

        entity.push(Entity {
            parent_id: Padded(parent_id),
        });
        for m in m_models.into_iter() {
            let (
                m_basevertex,
                m_entity,
                m_vertex_matrixtransformation,
                m_vertex_matrixtransformer,
                m_vertex_colortransformation,
                m_vertex_colortransformer,
                m_model_matrixtransformation,
                m_model_matrixtransformer
            ) = from_model(m, current_id, entity_index, vertex_matrixtransformation_offset, vertex_colortransformation_offset, model_matrixtransformation_offset);

            basevertex.extend(m_basevertex);
            entity.extend(m_entity);
            vertex_matrixtransformation.extend(m_vertex_matrixtransformation);
            vertex_matrixtransformer.extend(m_vertex_matrixtransformer);
            vertex_colortransformation.extend(m_vertex_colortransformation);
            vertex_colortransformer.extend(m_vertex_colortransformer);
            model_matrixtransformation.extend(m_model_matrixtransformation);
            model_matrixtransformer.extend(m_model_matrixtransformer);
        }

        (
            basevertex,
            entity,

            vertex_matrixtransformation,
            vertex_matrixtransformer,
            vertex_colortransformation,
            vertex_colortransformer,

            model_matrixtransformation,
            model_matrixtransformer,
        )
    }

    let mut vsinput_data: Vec<BaseVertex> = vec![];
    for vert in basevertex_data.iter() {
        vsinput_data.push((*vert).clone())
    }

    let dummy_vertex = vsinput_data[0];
    let dummy_mat_t: MatrixTransformer = vertex_matrixtransformer_data[0];
    let dummy_vec_t = vertex_colortransformer_data[0];
    
    let diff = 64-(vsinput_data.len()%64);
    let mut vertex_dispatch_len = basevertex_data.len() as u32 / 64;
    if diff!=64 {
        vertex_dispatch_len+=1;
        while vsinput_data.len()%64!=0 {
            vsinput_data.push(dummy_vertex);
            basevertex_data.push(dummy_vertex);
            vertex_matrixtransformer_data.push(dummy_mat_t);
            vertex_colortransformer_data.push(dummy_vec_t);
        }
    }

    let diff = 64-(model_matrixtransformer_data.len()%64);
    let mut model_dispatch_len = entity_data.len() as u32 / 64;
    if diff!=64 {
        model_dispatch_len+=1;
        while model_matrixtransformer_data.len()%64!=0 {
            model_matrixtransformer_data.push(dummy_mat_t);
        }
    }

    let modelt_data = vec![MatrixT {
        mat: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    }; entity_data.len()];

    debug_assert!(
        basevertex_data.len()%64 == 0 &&
        basevertex_data.len() > 0 &&
        basevertex_data.len() == vsinput_data.len() &&
        basevertex_data.len() == vertex_matrixtransformer_data.len() &&
        basevertex_data.len() == vertex_colortransformer_data.len() &&
        entity_data.len() > 0 &&
        entity_data.len() == (entity_index) as usize &&
        entity_data.len() <= model_matrixtransformer_data.len() &&
        model_matrixtransformer_data.len() < (entity_index) as usize + 64,
        "incoherent buffers lengths"
    );
    
    // BUFFERS

    // make non-empty
    if vertex_matrixtransformation_data.is_empty() { vertex_matrixtransformation_data.push(MatrixTransformation::default()); }
    if vertex_colortransformation_data.is_empty() { vertex_colortransformation_data.push(ColorTransformation::default()); }
    if model_matrixtransformation_data.is_empty() { model_matrixtransformation_data.push(MatrixTransformation::default()); }

    let len = vsinput_data.len() as u64;
    let vsinput_buffer: Subbuffer<[BaseVertex]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
        vsinput_data,
        len,
    );
    
    let len = basevertex_data.len() as u64;
    let basevertex_buffer: Subbuffer<[BaseVertex]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        basevertex_data,
        len,
    );
    
    let len = entity_data.len() as u64;
    let entity_buffer: Subbuffer<[Entity]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        entity_data,
        len,
    );
    
    let len = modelt_data.len() as u64;
    let modelt_buffer: Subbuffer<[MatrixT]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        modelt_data,
        len,
    );
    
    let len = vertex_matrixtransformation_data.len() as u64;
    let vertex_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_matrixtransformation_data,
        len,
    );
    
    let len = vertex_matrixtransformer_data.len() as u64;
    let vertex_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_matrixtransformer_data,
        len,
    );
    
    let len = vertex_colortransformation_data.len() as u64;
    let vertex_colortransformation_buffer: Subbuffer<[ColorTransformation]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_colortransformation_data,
        len,
    );
    
    let len = vertex_colortransformer_data.len() as u64;
    let vertex_colortransformer_buffer: Subbuffer<[ColorTransformer]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_colortransformer_data,
        len,
    );
    
    let len = model_matrixtransformation_data.len() as u64;
    let model_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        model_matrixtransformation_data,
        len,
    );
    
    let len = model_matrixtransformer_data.len() as u64;
    let model_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        model_matrixtransformer_data,
        len,
    );


    // Info

    dbg!("vsinput size: {}", vsinput_buffer.size());
    dbg!("basevertex size: {}", basevertex_buffer.size());
    dbg!("entity size: {}", entity_buffer.size());
    dbg!("modelt size: {}", modelt_buffer.size());
    dbg!("vertex_matrixtransformation size: {}", vertex_matrixtransformation_buffer.size());
    dbg!("vertex_matrixtransformer size: {}", vertex_matrixtransformer_buffer.size());
    dbg!("vertex_colortransformation size: {}", vertex_colortransformation_buffer.size());
    dbg!("vertex_colortransformer size: {}", vertex_colortransformer_buffer.size());
    dbg!("model_matrixtransformation size: {}", model_matrixtransformation_buffer.size());
    dbg!("model_matrixtransformer size: {}", model_matrixtransformer_buffer.size());
    dbg!("entity_index: {}", entity_index);
    dbg!("vertex_matrixtransformation_offset: {}", vertex_matrixtransformation_offset);
    dbg!("vertex_colortransformation_offset: {}", vertex_colortransformation_offset);
    dbg!("model_matrixtransformation_offset: {}", model_matrixtransformation_offset);

    (
        (input.meta.bg, input.meta.start, input.meta.end, vertex_dispatch_len, model_dispatch_len),
        (general_push_cs, general_push_vs, general_push_transformer),
        (
            vsinput_buffer,
            basevertex_buffer,
            entity_buffer,
            modelt_buffer,

            vertex_matrixtransformation_buffer,
            vertex_matrixtransformer_buffer,
            vertex_colortransformation_buffer,
            vertex_colortransformer_buffer,

            model_matrixtransformation_buffer,
            model_matrixtransformer_buffer,
        ),
    )
}

fn to_gpu_tf(t: Vec<Tf>) -> Vec<MatrixTransformation> {
    let mut gpu_tf: Vec<MatrixTransformation> = vec![];
    for tf in t.into_iter() {
        let (ty, val) = match *tf.read_t() {
            Transformation::Scale(x, y, z) => (0, [x, y, z]),
            Transformation::Translate(x, y, z) => (1, [x, y, z]),
            Transformation::RotateX(angle) => (2, [angle, 0.0, 0.0]),
            Transformation::RotateY(angle) => (3, [angle, 0.0, 0.0]),
            Transformation::RotateZ(angle) => (4, [angle, 0.0, 0.0]),
            _ => { println!("Vertex/Model transformation not implemented, ignoring."); continue; },
        };
        let evolution = match *tf.read_e() {
            Evolution::Linear => 0,
            Evolution::FastIn | Evolution::SlowOut => 1,
            Evolution::FastOut | Evolution::SlowIn => 2,
            Evolution::FastMiddle | Evolution::SlowInOut => 3,
            Evolution::FastInOut | Evolution::SlowMiddle =>  4,
        };
        gpu_tf.push(MatrixTransformation {
            val,
            ty,
            start: *tf.read_start(),
            end: *tf.read_end(),
            evolution: Padded(evolution),
        })
    }

    gpu_tf
}

fn to_gpu_cl(t: Vec<Cl>) -> Vec<ColorTransformation> {
    let mut gpu_cl: Vec<ColorTransformation> = vec![];
    for cl in t.into_iter() {
        let (ty, val) = match *cl.read_c() {
            Colorization::ToColor(r, g, b, a) => (0, [r, g, b, a]),
            // _ => { println!("Vertex colorization not implemented, ignoring."); continue; },
        };
        let evolution = match *cl.read_e() {
            Evolution::Linear => 0,
            Evolution::FastIn | Evolution::SlowOut => 1,
            Evolution::FastOut | Evolution::SlowIn => 2,
            Evolution::FastMiddle | Evolution::SlowInOut => 3,
            Evolution::FastInOut | Evolution::SlowMiddle =>  4,
        };
        gpu_cl.push(ColorTransformation {
            val,
            ty,
            start: *cl.read_start(),
            end: *cl.read_end(),
            evolution,
        })
    }

    gpu_cl
}

/// Creates a device-locel buffer with the given `len` and `iter` iterated data. `usage` must ocntain `BufferUsage::TRANSFER_DST`
fn create_buffer<T, I>(
    queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    cb_allocator: &StandardCommandBufferAllocator,
    usage: BufferUsage,
    iter: I,
    len: u64,
) -> Subbuffer<[T]>
where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    let buffer: Subbuffer<[T]> = Buffer::new_slice::<T>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        len,
    )
    .expect("failed to create buffer");

    let staging_buffer: Subbuffer<[T]> = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        iter,
    )
    .expect("failed to create staging_buffer");

    let mut cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
        AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create cbb");

    cbb
        .copy_buffer(CopyBufferInfo::buffers(
            staging_buffer.clone(),
            buffer.clone(),
        ))
        .unwrap();

    let copy_command_buffer: Arc<PrimaryAutoCommandBuffer> = cbb.build().unwrap();

    copy_command_buffer
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None /* timeout */)
        .unwrap();


    buffer
}
impl Vk {
    fn recreate_swapchain(&mut self) {
        let image_extent: [u32; 2] = self.window.inner_size().into();
        if image_extent.contains(&0) {
            return;
        }
        if self.recreate_swapchain {
            self.recreate_swapchain = false;

            (self.swapchain, self.images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            let extent = self.images[0].extent();

            let msaa_image: Arc<ImageView> = ImageView::new_default(
                Image::new(
                    self.memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: self.swapchain.image_format(),
                        extent,
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                        samples: SampleCount::Sample8,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();

            let msaa_depth_attachment = ImageView::new_default(
                Image::new(
                    self.memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D16_UNORM,
                        extent,
                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                        samples: SampleCount::Sample8,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();

            self.framebuffers = self
                .images
                .iter()
                .map(|image| {
                    let final_view = ImageView::new_default(image.clone()).unwrap();
                    Framebuffer::new(
                        self.render_pass.clone(),
                        FramebufferCreateInfo {
                            attachments: vec![msaa_image.clone(), final_view, msaa_depth_attachment.clone()],
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

            self.draw_pipeline = {
                let vertex_input_state = BaseVertex::per_vertex()
                    .definition(&self.drawing_vs.info().input_interface)
                    .unwrap();
                let stages = [
                    PipelineShaderStageCreateInfo::new(self.drawing_vs.clone()),
                    PipelineShaderStageCreateInfo::new(self.drawing_fs.clone()),
                ];
                let layout = PipelineLayout::new(
                    self.device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                        .into_pipeline_layout_create_info(self.device.clone())
                        .unwrap(),
                )
                .unwrap();
                let subpass = Subpass::from(self.render_pass.clone(), 0).unwrap();

                GraphicsPipeline::new(
                    self.device.clone(),
                    None,
                    GraphicsPipelineCreateInfo {
                        stages: stages.into_iter().collect(),
                        vertex_input_state: Some(vertex_input_state),
                        input_assembly_state: Some(InputAssemblyState::default()),
                        viewport_state: Some(ViewportState {
                            viewports: [Viewport {
                                offset: [0.0, 0.0],
                                extent: [extent[0] as f32, extent[1] as f32],
                                depth_range: 0.0..=1.0,
                            }]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        }),
                        rasterization_state: Some(RasterizationState::default()),
                        multisample_state: Some(MultisampleState {
                            rasterization_samples: subpass.num_samples().unwrap(),
                            ..Default::default()
                        }),
                        color_blend_state: Some(ColorBlendState::with_attachment_states(
                            subpass.num_color_attachments(),
                            ColorBlendAttachmentState {
                                blend: Some(AttachmentBlend::alpha()),
                                ..Default::default()
                            },
                        )),
                        subpass: Some(subpass.into()),
                        depth_stencil_state: Some(DepthStencilState {
                            depth: Some(DepthState::simple()),
                            ..Default::default()
                        }),
                        ..GraphicsPipelineCreateInfo::layout(layout)
                    },
                )
                .unwrap()
            };
        }
    }

    /// Cleans resources, updates buffers and draws a frame.
    fn draw(&mut self) {
        let image_extent: [u32; 2] = self.window.inner_size().into();
        if image_extent.contains(&0) {
            return;
        }
        self.recreate_swapchain();

        let (image_index, suboptimal, acquire_future) = match acquire_next_image(
            self.swapchain.clone(),
            None, // timeout
        ) {
            Ok(tuple) => tuple,
            Err(e) => panic!("failed to acquire next image: {e}"),
        };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        if let Some(image_fence) = &mut self.drawing_fences[image_index as usize] {
            image_fence.cleanup_finished();
        }

        let previous_future = match self.drawing_fences[self.previous_drawing_fence_i as usize].take() {
            Some(fence) => fence.boxed(),
            None => sync::now(self.device.clone()).boxed(),
        };


        let mut mat: Mat4 = self.general_push_transformer.0.update_vp(self.time);
        mat.mult(self.general_push_transformer.1.update_vp(self.time));
        mat.mult(Mat4::scale((image_extent[1] as f32 / image_extent[0] as f32).min(1.0), (image_extent[0] as f32 / image_extent[1] as f32).min(1.0), 1.0));
        self.general_push_vs = VSGeneral {
            mat: mat.0,
        };


        let mut builder = AutoCommandBufferBuilder::primary(
            &self.cb_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .push_constants(self.vertex_compute_pipeline.layout().clone(), 0, self.general_push_cs.clone())
            .unwrap()
            .bind_pipeline_compute(self.vertex_compute_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.vertex_compute_pipeline.layout().clone(),
                0,
                self.vertex_compute_descriptor_set.clone(),
            )
            .unwrap()
            .dispatch([self.vertex_dispatch_len, 1, 1])
            .unwrap()
            .push_constants(self.model_compute_pipeline.layout().clone(), 0, self.general_push_cs.clone())
            .unwrap()
            .bind_pipeline_compute(self.model_compute_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.model_compute_pipeline.layout().clone(),
                0,
                self.model_compute_descriptor_set.clone(),
            )
            .unwrap()
            .dispatch([self.model_dispatch_len, 1, 1])
            .unwrap()
            // Compute -> Graphics
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some(self.background_color.into()),
                        None,
                        Some(1.0.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index as usize].clone())
                },
                Default::default(), //vulkano::command_buffer::SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
            )
            .unwrap()
            .push_constants(self.draw_pipeline.layout().clone(), 0, self.general_push_vs.clone())
            .unwrap()
            .bind_pipeline_graphics(self.draw_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.draw_pipeline.layout().clone(),
                0,
                self.draw_descriptor_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vsinput_buffer.clone())
            .unwrap()
            .draw(self.vsinput_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass(Default::default())
            .unwrap()
            ;

        let command_buffer = builder.build().unwrap();

        let future = previous_future
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        self.drawing_fences[image_index as usize] = match future.map_err(Validated::unwrap) {
            Ok(future) => Some(future),
            Err(e) => panic!("failed to flush future: {e}"),
        };
        self.previous_drawing_fence_i = image_index;
    }

    /// Runs the animation in the window in real-time.
    fn show(
        &mut self,
        event_loop: &mut EventLoop<()>,
        // save: (u32, u32),
    ) -> i32 {
        println!("\n♻ --- {}: Showing with updated data.\n", self.show_count);
        let mut max_elapsed = true;
        let start = Instant::now();
        event_loop.run_return(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::ExitWithCode(1);
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                self.recreate_swapchain = true;
            }
            Event::MainEventsCleared => {
                if max_elapsed {
                    self.recreate_swapchain = true;         // TODOFEATURES
                    self.time = start.elapsed().as_secs_f32() + self.start_time;
                    self.general_push_cs = CSGeneral { time: self.time };
                    // if elements.ended() {
                    //     *control_flow = ControlFlow::ExitWithCode(0);
                    // }
                    if self.time > self.end_time {
                            max_elapsed = false;
                            self.show_count += 1;
                        *control_flow = ControlFlow::ExitWithCode(0);
                    }
                    self.draw();
                    // if save.0 > 0 && save.0 > 0 { self.encode(); }
                }
            }
            _ => (),
        })
    }

    // /// Encodes a frame to the output video, for saving.
    // fn encode(&mut self) {
    //     unimplemented!("Cannot encode yet!");
    // }
}
