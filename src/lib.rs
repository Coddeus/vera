pub mod elements;
pub use elements::*;
pub mod update;
pub use update::*;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, GenericMemoryAllocator, MemoryUsage, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

pub struct Vera {
    // -----
    library: Arc<vulkano::VulkanLibrary>,
    required_extensions: vulkano::instance::InstanceExtensions,
    instance: Arc<Instance>,
    event_loop: EventLoop<()>,
    surface: Arc<Surface>,
    window: Arc<Window>,

    device_extensions: DeviceExtensions,
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,

    swapchain: Arc<Swapchain>,
    images: Vec<Arc<SwapchainImage>>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,

    // -----
    memory_allocator: GenericMemoryAllocator<Arc<vulkano::memory::allocator::FreeListAllocator>>,
    command_buffer_allocator: StandardCommandBufferAllocator,

    // -----
    max_uniform_buffer_size: u32,
    max_storage_buffer_size: u32,

    // -----
    vertex_buffer: Subbuffer<[Veratex]>,

    // -----
    staging_uniform_buffer: Subbuffer<[UniformData]>,
    uniform_buffer: Subbuffer<[UniformData]>,
    uniform_update_cs: Arc<ShaderModule>,
    uniform_update_pipeline: Arc<ComputePipeline>,
    uniform_copy_command_buffer: Arc<PrimaryAutoCommandBuffer>,
    uniform_update_command_buffer: Arc<PrimaryAutoCommandBuffer>,

    // -----
    drawing_vs: Arc<ShaderModule>,
    drawing_fs: Arc<ShaderModule>,
    drawing_viewport: Viewport,
    drawing_pipeline: Arc<GraphicsPipeline>,
    drawing_command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,

    // -----
    window_resized: bool, // Should not be possible
    recreate_swapchain: bool,
    frames_in_flight: usize,
    previous_fence_i: u32,
}

const PKG_NAME: &str = match option_env!("CARGO_PKG_NAME") {
    Some(T) => T,
    None => "Vera",
};

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460
            
            layout(set = 0, binding = 0) buffer uData {
                uint data[];
            } buf;

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
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

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}

impl Vera {
    /// Sets up Vulkan for Vera
    pub fn create(width: u32, height: u32) -> Self {
        // Extensions/instance/event_loop/surface/window/physical_device/queue_family/device/queue/swapchain/images/render_pass/framebuffers
        // ---------------------------------------------------------------------------------------------------------------------------------
        let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = vulkano_win::required_extensions(&library);
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                application_name: Some("Vera".to_owned()),
                application_version: Version::major_minor(0, 1),
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .with_inner_size(LogicalSize { width, height })
            .with_resizable(false)
            .with_title(PKG_NAME)
            .with_transparent(true)
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
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
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
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

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("failed to get surface capabilities");

            let dimensions = window.inner_size();
            let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
            let image_format = Some(
                physical_device
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format,
                    image_extent: dimensions.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::single_pass_renderpass!(
            device,
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(), // set the format the same as the swapchain
                    samples: 1, // TODOSAMPLES
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // ---------------------------------------------------------------------------------------------------------------------------------

        // Allocators
        // ----------
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        // ----------

        // Max buffer sizes
        // ----------------
        // // If the elements fit, use UBO, otherwise use SSBO
        let max_uniform_buffer_size: u32 = physical_device.properties().max_uniform_buffer_range;
        let max_storage_buffer_size: u32 = physical_device.properties().max_storage_buffer_range;

        // ----------------

        // One-time copy of vertex data to device-local memory
        // ---------------------------------------------------
        let vertex_data = vec![
            Veratex::new(0.0, 0.0, 0),
            Veratex::new(1.0, 0.0, 0),
            Veratex::new(0.0, 1.0, 0),
        ]
        .into_iter();

        let temporary_vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                // Specify this buffer will be used as a transfer source.
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify this buffer will be used for uploading to the GPU.
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertex_data,
        )
        .expect("failed to create temporary_vertex_buffer");

        let vertex_buffer = Buffer::new_slice::<Veratex>(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            vertex_data.len() as u64,
        )
        .expect("failed to create vertex_buffer");

        let mut vertex_cbb = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create vertex_cbb");

        vertex_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                temporary_vertex_buffer,
                vertex_buffer.clone(),
            ))
            .unwrap();

        let vertex_cb = vertex_cbb.build().unwrap();
        vertex_cb
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        // ---------------------------------------------------

        // Staging & Device-local uniform buffers, and their copy & update command buffers
        // -------------------------------------------------------------------------------
        let uniform_data = vec![UniformData::empty(); 1];

        let staging_uniform_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            uniform_data,
        )
        .expect("failed to create staging_uniform_buffer");

        let uniform_buffer = Buffer::new_slice(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            uniform_data.len() as u64,
        )
        .expect("failed to create uniform_buffer");

        let mut uniform_copy_cbb = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create uniform_copy_cbb");

        uniform_copy_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                staging_uniform_buffer,
                uniform_buffer.clone(),
            ))
            .unwrap();

        let uniform_copy_command_buffer = Arc::new(uniform_copy_cbb.build().unwrap());
        uniform_copy_command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        let uniform_update_cs =
            cs::load(device.clone()).expect("failed to create compute shader module");

        let uniform_update_pipeline = ComputePipeline::new(
            device.clone(),
            uniform_update_cs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("failed to create compute pipeline");

        let mut uniform_update_cbb = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create uniform_update_cbb");

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let pipeline_layout = uniform_update_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [WriteDescriptorSet::buffer(
                0,
                staging_uniform_buffer.clone(),
            )], // 0 is the binding
        )
        .unwrap();

        uniform_update_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                staging_uniform_buffer,
                uniform_buffer.clone(),
            ))
            .unwrap();

        let uniform_update_command_buffer = Arc::new(uniform_update_cbb.build().unwrap());
        uniform_update_command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        // -------------------------------------------------------------------------------

        // Graphics pipeline & Drawing command buffer
        // ------------------------------------------

        let drawing_vs = vs::load(device.clone()).expect("failed to create vertex shader module");
        let drawing_fs = fs::load(device.clone()).expect("failed to create fragment shader module");

        let drawing_viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: window.inner_size().into(),
            depth_range: 0.0..1.0,
        };

        let drawing_pipeline = GraphicsPipeline::start()
            .vertex_input_state(Veratex::per_vertex())
            .vertex_shader(drawing_vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                drawing_viewport,
            ]))
            .fragment_shader(drawing_fs.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(device)
            .unwrap();

        // Command buffers:
        // 1. Compute update staging_uniform_buffer,                                     //
        // 1. Draw graphics pipeline using vertex_buffer and final_uniform_buffer,       //
        // 2. Copy data from staging_uniform_buffer to final_uniform_buffer,             //
        // 2. Swap swapchain images,                                                     // Done

        let drawing_command_buffers = framebuffers
            .iter()
            .map(|framebuffer| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 0.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .bind_pipeline_graphics(drawing_pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                Arc::new(builder.build().unwrap())
            })
            .collect();

        // ------------------------------------------
        Vera {
            //
            library,
            required_extensions,
            instance,
            event_loop,
            surface,
            window,

            device_extensions,
            physical_device,
            queue_family_index,
            device,
            queue,

            swapchain,
            images,
            render_pass,
            framebuffers,

            // -----
            memory_allocator,
            command_buffer_allocator,

            // -----
            max_uniform_buffer_size,
            max_storage_buffer_size,

            // -----
            vertex_buffer,

            // -----
            staging_uniform_buffer,
            uniform_buffer,
            uniform_update_cs,
            uniform_update_pipeline,
            uniform_copy_command_buffer,
            uniform_update_command_buffer,

            // -----
            drawing_vs,
            drawing_fs,
            drawing_viewport,
            drawing_pipeline,
            drawing_command_buffers,

            // -----
            window_resized: false,
            recreate_swapchain: false,
            frames_in_flight: 0,
            previous_fence_i: 0,
        }
    }

    pub fn show(mut self) {
        let mut window_resized = false;
        let mut recreate_swapchain = false;

        self.frames_in_flight = self.images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; self.frames_in_flight];
        self.previous_fence_i = 0;

        self.event_loop
            .run(move |event, _, control_flow| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    window_resized = true;
                }
                Event::MainEventsCleared => {
                    if window_resized || recreate_swapchain {
                        recreate_swapchain = false;

                        let new_dimensions = self.window.inner_size();

                        let (new_swapchain, new_images) =
                            match self.swapchain.recreate(SwapchainCreateInfo {
                                image_extent: new_dimensions.into(),
                                ..self.swapchain.create_info()
                            }) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                    return
                                }
                                Err(e) => panic!("failed to recreate swapchain: {e}"),
                            };
                        self.swapchain = new_swapchain;
                        let new_framebuffers = new_images
                            .iter()
                            .map(|image| {
                                let view = ImageView::new_default(image.clone()).unwrap();
                                Framebuffer::new(
                                    self.render_pass.clone(),
                                    FramebufferCreateInfo {
                                        attachments: vec![view],
                                        ..Default::default()
                                    },
                                )
                                .unwrap()
                            })
                            .collect::<Vec<_>>();

                        if window_resized {
                            window_resized = false;

                            self.drawing_viewport.dimensions = new_dimensions.into();

                            let new_pipeline = GraphicsPipeline::start()
                                .vertex_input_state(Veratex::per_vertex())
                                .vertex_shader(self.drawing_vs.entry_point("main").unwrap(), ())
                                .input_assembly_state(InputAssemblyState::new())
                                .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                                    self.drawing_viewport,
                                ]))
                                .fragment_shader(self.drawing_fs.entry_point("main").unwrap(), ())
                                .render_pass(Subpass::from(self.render_pass, 0).unwrap())
                                .build(self.device)
                                .unwrap();

                            self.drawing_command_buffers = new_framebuffers
                                .iter()
                                .map(|framebuffer| {
                                    let mut builder = AutoCommandBufferBuilder::primary(
                                        &self.command_buffer_allocator,
                                        self.queue.queue_family_index(),
                                        CommandBufferUsage::MultipleSubmit,
                                    )
                                    .unwrap();

                                    builder
                                        .begin_render_pass(
                                            RenderPassBeginInfo {
                                                clear_values: vec![Some(
                                                    [0.0, 0.0, 0.0, 0.0].into(),
                                                )],
                                                ..RenderPassBeginInfo::framebuffer(
                                                    framebuffer.clone(),
                                                )
                                            },
                                            SubpassContents::Inline,
                                        )
                                        .unwrap()
                                        .bind_pipeline_graphics(new_pipeline.clone())
                                        .bind_vertex_buffers(0, self.vertex_buffer.clone())
                                        .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                                        .unwrap()
                                        .end_render_pass()
                                        .unwrap();

                                    Arc::new(builder.build().unwrap())
                                })
                                .collect()
                        }
                    }

                    let (image_i, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("failed to acquire next image: {e}"),
                        };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    // wait for the fence related to this image to finish (normally this would be the oldest fence)
                    if let Some(image_fence) = &fences[image_i as usize] {
                        image_fence.wait(None).unwrap();
                    }

                    let previous_future = match fences[self.previous_fence_i as usize].clone() {
                        // Create a NowFuture
                        None => {
                            let mut now = sync::now(self.device.clone());
                            now.cleanup_finished();

                            now.boxed()
                        }
                        // Use the existing FenceSignalFuture
                        Some(fence) => fence.boxed(),
                    };

                    let future = previous_future
                        .join(acquire_future)
                        .then_execute(
                            self.queue.clone(),
                            self.drawing_command_buffers[image_i as usize].clone(),
                        )
                        .unwrap()
                        .then_swapchain_present(
                            self.queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                self.swapchain.clone(),
                                image_i,
                            ),
                        )
                        .then_signal_fence_and_flush();

                    fences[image_i as usize] = match future {
                        Ok(value) => Some(Arc::new(value)),
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            None
                        }
                        Err(e) => {
                            println!("failed to flush future: {e}");
                            None
                        }
                    };

                    self.previous_fence_i = image_i;
                }
                _ => (),
            });
    }
}
