#![deny(clippy::all)]
//#![forbid(unsafe_code)]

use log::{debug, error};
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalSize};
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit_input_helper::WinitInputHelper;
use ggez::nalgebra as na;
use na::{U2, U3, Dynamic, Matrix, MatrixArray, MatrixVec, Vector2, Matrix4};
use typenum::U200;
use std::time::{Duration, SystemTime};
use std::thread::sleep;


//type MyMatrix = Matrix<f64, U200, U200, MatrixArray<f64, U200, U200>>;
type MyMatrix = Matrix<f64, Dynamic, Dynamic, MatrixVec<f64, Dynamic, Dynamic>>;
type MyMatrix4 = Matrix<u32, Dynamic, Dynamic, MatrixVec<u32, Dynamic, Dynamic>>;
type BoolMatrix = Matrix<bool, Dynamic, Dynamic, MatrixVec<bool, Dynamic, Dynamic>>;

const C: f64 = 1.0;
const W: f64 = 1.9;
const ROWS: usize = 400;
const COLUMNS: usize = 400;

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let (window, surface, p_width, p_height, mut hidpi_factor) =
        create_window("flud", &event_loop);

    let surface_texture = SurfaceTexture::new(p_width, p_height, surface);

    let mut fluid = Fluid::new();
    let mut pixels = Pixels::new(COLUMNS as u32, ROWS as u32, surface_texture)?;
    let mut paused = false;

    let mut draw_state: Option<bool> = None;

    event_loop.run(move |event, _, control_flow| {
        // The one and only event that winit_input_helper doesn't have for us...
        if let Event::RedrawRequested(_) = event {
            fluid.draw(pixels.get_frame());
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // For everything else, for let winit_input_helper collect events to build its state.
        // It returns `true` when it is time to update our game state and request a redraw.
        if input.update(event) {
            let (mouse_cell, mouse_prev_cell) = input
            .mouse()
            .map(|(mx, my)| {
                let (dx, dy) = input.mouse_diff();
                let prev_x = mx - dx;
                let prev_y = my - dy;
                let dpx = 2.0;//hidpi_factor as f32;
                let (w, h) = (p_width as f32 / dpx, p_height as f32 / dpx);
                let mx_i = ((mx / w) * (COLUMNS as f32)).round() as isize;
                let my_i = ((my / h) * (ROWS as f32)).round() as isize;
                let px_i = ((prev_x / w) * (COLUMNS as f32)).round() as isize;
                let py_i = ((prev_y / h) * (ROWS as f32)).round() as isize;
                ((mx_i, my_i), (px_i, py_i))
            })
            .unwrap_or_default();

            if input.mouse_pressed(0) {
                debug!("Mouse click at {:?}", mouse_cell);
                draw_state = Some(fluid.toggle(mouse_cell.0, mouse_cell.1));
            } else if let Some(draw_obstacle) = draw_state {
                let release = input.mouse_released(0);
                let held = input.mouse_held(0);
                debug!("Draw at {:?} => {:?}", mouse_prev_cell, mouse_cell);
                debug!("Mouse held {:?}, release {:?}", held, release);
                // If they either released (finishing the drawing) or are still
                // in the middle of drawing, keep going.
                if release || held {
                    debug!("Draw line of {:?}", draw_obstacle);
                    fluid.set_line(
                        mouse_prev_cell.0,
                        mouse_prev_cell.1,
                        mouse_cell.0,
                        mouse_cell.1,
                    );
                }
                // If they let go or are otherwise not clicking anymore, stop drawing.
                if release || !held {
                    debug!("Draw end");
                    draw_state = None;
                }
            }
            // Close events
            // Adjust high DPI factor
            if let Some(factor) = input.scale_factor_changed() {
                hidpi_factor = factor;
            }
            // Resize the window
            if let Some(size) = input.window_resized() {
                pixels.resize(size.width, size.height);
            }
            if !paused {
                let now = SystemTime::now();
                fluid.update_cell_macro();
                println!("update_cell_macro(): {}", now.elapsed().unwrap().as_micros());
                let now = SystemTime::now();
                fluid.new_density();
                println!("new_density(): {}", now.elapsed().unwrap().as_micros());
                let now = SystemTime::now();
                fluid.stream();
                println!("stream: {}", now.elapsed().unwrap().as_micros());
                fluid.count += 1;
            }
            window.request_redraw();
        }
    });
}

struct ColourMap {
    red_list: Vec<u32>,
    green_list: Vec<u32>,
    blue_list: Vec<u32>,
}

impl ColourMap {
    pub fn gen_jet_colourmap() -> ColourMap {
        let n_colours: usize = 400;							// there are actually n_colours+2 colors
    
        let mut red_list: Vec<u32> = Vec::with_capacity(n_colours+2);
        let mut green_list: Vec<u32> = Vec::with_capacity(n_colours+2);
        let mut blue_list: Vec<u32> = Vec::with_capacity(n_colours+2);
        for i in 0..n_colours {
            let r: u32;
            let g: u32;
            let b: u32;
    
            if (i as f64) < n_colours as f64 / 8.0 {
                r = 0; 
                g = 0; 
                b = (255.0 * (i as f64 + n_colours as f64 / 8.0) / (n_colours as f64 / 4.0)) as u32;
            } else if (i as f64) < 3.0 * n_colours as f64 / 8.0 {
                r = 0;
                g = (255.0 * (i as f64 - n_colours as f64 / 8.0) / (n_colours as f64 / 4.0)) as u32; 
                b = 255;
            } else if (i as f64) < (5.0 * n_colours as f64 / 8.0) {
                r = (255.0 * (i as f64 - 3.0 * n_colours as f64 / 8.0) / (n_colours as f64 / 4.0)) as u32; 
                g = 255; 
                b = 255 - r;
            } else if (i as f64) < 7.0 * n_colours as f64 / 8.0 {
                r = 255; 
                g = (255.0 * (7.0 * n_colours as f64 / 8.0 - i as f64) / (n_colours as f64 / 4.0)) as u32; 
                b = 0;
            } else {
                r = (255.0 * (9.0 * n_colours as f64 / 8.0 - i as f64) / (n_colours as f64 / 4.0)) as u32; 
                g = 0; 
                b = 0;
            }
    
            red_list.push(r); 
            green_list.push(g); 
            blue_list.push(b);
            //println!("{}", r);
        }
        
        ColourMap {
            red_list,
            green_list,
            blue_list,
        }
    }
}

#[derive(Clone, Debug)]
struct TotalVelocity {
    x: MyMatrix,
    y: MyMatrix,
}

impl TotalVelocity {
    fn new(velocity: (f64, f64)) -> TotalVelocity {
        TotalVelocity {
            x: MyMatrix::repeat(ROWS, COLUMNS, velocity.0),
            y: MyMatrix::repeat(ROWS, COLUMNS, velocity.1),
        }
    }

    fn sq_hypot(&self) -> MyMatrix {
        
        let x_square = self.x.component_mul(&self.x);
        let y_square = self.y.component_mul(&self.y);
        
        x_square + y_square
    }

}

#[derive(Clone, Debug)]
struct CellConsts {
    prob: f64,
    disp: Vector2<f64>,
}

impl CellConsts {
    pub fn get_consts() -> Vec<CellConsts> {
        let mut cell_consts = vec![];
        cell_consts.push(CellConsts {
            prob: 4.0 / 9.0,
            disp: Vector2::new(0.0, 0.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 9.0,
            disp: Vector2::new(1.0, 0.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 9.0,
            disp: Vector2::new(0.0, 1.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 9.0,
            disp: Vector2::new(-1.0, 0.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 9.0,
            disp: Vector2::new(0.0, -1.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 36.0,
            disp: Vector2::new(1.0, 1.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 36.0,
            disp: Vector2::new(-1.0, 1.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 36.0,
            disp: Vector2::new(-1.0, -1.0),
        });
        cell_consts.push(CellConsts {
            prob: 1.0 / 36.0,
            disp: Vector2::new(1.0, -1.0),
        });

        cell_consts
    }
}

struct Fluid {
    rows: usize,
    columns: usize,
    norm_density: Vec<MyMatrix>,
    eq_density: Vec<MyMatrix>,
    obstacles: MyMatrix,
    total_velocity: TotalVelocity,
    total_density: MyMatrix,
    cell_consts: Vec<CellConsts>,
    colour_map: ColourMap,
    count: usize,
    ones: MyMatrix,
}

impl Fluid {
    pub fn new() -> Fluid {
        // Load/create resources such as images here.
        Fluid {
            // Your state here...
            // densities starting at vector e0
            rows: ROWS,
            columns: COLUMNS,
            norm_density: Fluid::initialise_density((0.1, 0.0)),
            eq_density: Fluid::density(ROWS, COLUMNS),
            obstacles: Fluid::init_obstacles(),
            total_velocity: TotalVelocity::new((0.1, 0.0)),
            total_density: MyMatrix::zeros(ROWS, COLUMNS),
            cell_consts: CellConsts::get_consts(),
            colour_map: ColourMap::gen_jet_colourmap(),
            count: 0,
            ones: MyMatrix::repeat(ROWS, COLUMNS, 1.0),
        }
    }  

    pub fn init_obstacles() -> MyMatrix {
        let mut new_obstacles = MyMatrix::zeros(ROWS, COLUMNS);
        //new_obstacles[(10, 10)] = 1.0;
        //new_obstacles[(11, 10)] = 1.0;
        //new_obstacles[(12, 10)] = 1.0;
        //new_obstacles[(13, 10)] = 1.0;
        //new_obstacles[(14, 10)] = 1.0;
        //new_obstacles[(15, 10)] = 1.0;
        new_obstacles
    }

    pub fn set_line(&mut self, x0: isize, y0: isize, x1: isize, y1: isize) {
        // probably should do sutherland-hodgeman if this were more serious.
        // instead just clamp the start pos, and draw until moving towards the
        // end pos takes us out of bounds.
        let x0 = x0.max(0).min(COLUMNS as isize);
        let y0 = y0.max(0).min(ROWS as isize);
        for (x, y) in line_drawing::Bresenham::new((x0, y0), (x1, y1)) {
            if let Some((x, y)) = self.grid_idx(x, y) {
                if self.obstacles[(y, x)] == 1.0 {
                    self.obstacles[(y, x)] = 0.0;
                } else {
                    self.obstacles[(y, x)] = 1.0;
                }
            } else {
                break;
            }
        }
    }

    pub fn toggle(&mut self, x: isize, y: isize) -> bool {
        if let Some((x, y)) = self.grid_idx(x, y) {
            if self.obstacles[(y, x)] == 1.0 {
                self.obstacles[(y, x)] = 0.0;
            } else {
                self.obstacles[(y, x)] = 1.0;
            }
        }
        true
    }

    pub fn grid_idx<T: std::convert::TryInto<usize>>(&self, x: T, y: T) -> Option<(usize, usize)> {
        if let (Ok(x), Ok(y)) = (x.try_into(), y.try_into()) {
            if x < COLUMNS - 1 && y < ROWS - 1{
                Some((x, y))
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn sum_densities(&self, density: &Vec<MyMatrix>) -> MyMatrix {
        let mut sum = MyMatrix::zeros(ROWS, COLUMNS);
        for i in 0..9 {
            sum += &density[i];
        }
        sum
    }

    pub fn density(rows: usize, columns: usize) -> Vec<MyMatrix> {
        let mut density_vec = vec![];

        for _ in 0..9 {
            let mut matrix = MyMatrix::repeat(ROWS, COLUMNS, 1.0);
            density_vec.push(matrix);
        }

        density_vec
    }

    pub fn update_cell_macro(&mut self) {
        //println!("{:?}", self.norm_density);
        self.total_density = self.sum_densities(&self.norm_density);

        self.total_velocity.x = (&self.norm_density[1] -
        &self.norm_density[3] +
        &self.norm_density[5] -
        &self.norm_density[6] - 
        &self.norm_density[7] +
        &self.norm_density[8]).component_div(&self.total_density);
        //println!("{:?}", self.total_velocity.x);

        self.total_velocity.y = (&self.norm_density[2] -
        &self.norm_density[4] +
        &self.norm_density[5] +
        &self.norm_density[6] -
        &self.norm_density[7] -
        &self.norm_density[8]).component_div(&self.total_density);
    }

    pub fn initialise_density(velocity: (f64, f64)) -> Vec<MyMatrix> {
        let total_velocity = TotalVelocity::new(velocity);
        let cell_consts = CellConsts::get_consts();
        let mut density_vec = vec![];

        let pre_calculated = - 3.0 / 2.0 * total_velocity.sq_hypot();

        //println!("{:?}", pre_calculated);

        for i in 0..9 {
            let dot_prod = &total_velocity.x * cell_consts[i].disp[0] + &total_velocity.y * cell_consts[i].disp[1];
            let dot_prod_square = &dot_prod.component_mul(&dot_prod);
            let eq_density = cell_consts[i].prob * ((MyMatrix::repeat(ROWS, COLUMNS, 1.0) + 3.0 * dot_prod) + 
            4.5 * dot_prod_square + &pre_calculated);
            density_vec.push(eq_density);
        }
        density_vec
    }

    pub fn new_density(&mut self) {
        let pre_calculated = self.total_velocity.x.zip_map(&self.total_velocity.y, |x, y| - 3.0 / 2.0 * (x * x + y * y));

        for i in 0..9 {
            let dot_prod =  self.total_velocity.x.zip_map(&self.total_velocity.y, |x, y| x * self.cell_consts[i].disp[0] + y * self.cell_consts[i].disp[1]);

            self.eq_density[i] = self.total_density.zip_zip_map(&dot_prod, &pre_calculated, |tot_density, dot, pre_calc| tot_density * self.cell_consts[i].prob * (1.0 + 3.0 * dot + 4.5 * dot * dot + pre_calc));

            self.norm_density[i].zip_apply(&self.eq_density[i], |norm, eq| norm + W * (eq - norm));
        }

    }

    pub fn set_column(&mut self, col: usize, index: usize, new_density: &Vec<MyMatrix>) {
        for m in 0..ROWS {
            self.norm_density[index][(m, col)] = new_density[index][(m, col)];
        }
    }

    pub fn roll(matrix: &MyMatrix, shift: i32, axis: u32) -> MyMatrix {
        //axis 0 is x axis, 1 y axis
        let mut new = MyMatrix::zeros(ROWS, COLUMNS);
        new.copy_from(&matrix);

        match axis {
            0 if shift > 0 => {
                new = new.insert_columns(0, 1, 0.0);
                new.set_column(0, &matrix.column(matrix.ncols() - 1));
                new = new.remove_column(matrix.ncols());
            },
            0 if shift < 0 => {
                new = new.insert_columns(matrix.ncols(), 1, 0.0);
                new.set_column(matrix.ncols(), &matrix.column(0));
                new = new.remove_column(0);
            },
            1 if shift > 0 => {
                new = new.insert_rows(matrix.nrows(), 1, 0.0);
                new.set_row(matrix.nrows(), &matrix.row(0));
                new = new.remove_row(0);
            },
            1 if shift < 0 => {
                new = new.insert_rows(0, 1, 0.0);
                new.set_row(0, &matrix.row(matrix.nrows() - 1));
                new = new.remove_row(matrix.nrows());
            },
            _ => (),
        }
        new
    }


    pub fn stream(&mut self) {
        let mut new_density = Fluid::initialise_density((0.1, 0.0));

        self.norm_density[1] = Fluid::roll(&self.norm_density[1], 1, 0);
        self.norm_density[2] = Fluid::roll(&self.norm_density[2], 1, 1);
        self.norm_density[3] = Fluid::roll(&self.norm_density[3], -1, 0);
        self.norm_density[4] = Fluid::roll(&self.norm_density[4], -1, 1);

        self.norm_density[5] = Fluid::roll(&self.norm_density[5], 1, 0);
        self.norm_density[5] = Fluid::roll(&self.norm_density[5], 1, 1);

        self.norm_density[6] = Fluid::roll(&self.norm_density[6], -1, 0);
        self.norm_density[6] = Fluid::roll(&self.norm_density[6], 1, 1);

        self.norm_density[7] = Fluid::roll(&self.norm_density[7], -1, 0);
        self.norm_density[7] = Fluid::roll(&self.norm_density[7], -1, 1);

        self.norm_density[8] = Fluid::roll(&self.norm_density[8], 1, 0);
        self.norm_density[8] = Fluid::roll(&self.norm_density[8], -1, 1);

        
        self.set_column(0, 1, &new_density);
        self.set_column(0, 3, &new_density);
        self.set_column(0, 5, &new_density);
        self.set_column(0, 6, &new_density);
        self.set_column(0, 7, &new_density);
        self.set_column(0, 8, &new_density);

        //println!("{:?}", self.norm_density[8]);
        for m in 0..ROWS {
            for n in 0..COLUMNS {
                if self.obstacles[(m, n)] == 1.0 {
                    for i in 1..9 {
                        let coords = ((m as f64 - self.cell_consts[i].disp[1]) as usize, (n as f64 + self.cell_consts[i].disp[0]) as usize);
                        let m = m as usize;
                        let n = n as usize;

                        if self.obstacles[coords] == 0.0 {
                            //println!("hi");
                            let other_index: usize;
                            if i <= 4 {
                                other_index = (&i - 1 + 2) % 4 + 1;   
                            } else {
                                other_index = (&i - 5 + 2) % 4 + 5;
                            }
                            //println!("{} {}", i, other_index);
                            self.norm_density[i][coords] = self.norm_density[other_index][(m, n)];
                        }
                    }
                }
            }
        }
    }

    pub fn draw(&self, screen: &mut [u8]) {
        let mut screen_iter = screen.chunks_exact_mut(4);

        let now = SystemTime::now();
        // Draw code here...
        
        let width = 5.0;

        let v_max = 0.4;

        for m in 0..ROWS {
            for n in 0..COLUMNS {
                let mut pix = screen_iter.next().unwrap();
                
                let velocity = (self.total_velocity.x[(m, n)].powi(2) + self.total_velocity.y[(m, n)].powi(2)).sqrt();
                let mut index = (velocity * 6.0 * 400.0) as usize;

                if index > 399 {
                    index = 399;
                } else if index < 0 {
                    index = 0;
                };

                let r = self.colour_map.red_list[index] as u8;
                let g = self.colour_map.green_list[index] as u8;
                let b = self.colour_map.blue_list[index] as u8;

                let cell_colour: [u8; 4];

                if self.obstacles[(m, n)] == 1.0 {
                    cell_colour = [0, 0, 0, 1];
                } else {
                    cell_colour = [r, g, b, 1];
                }
                //let cell_colour = [0, 0xff, 0xff, 0xff];
                pix.copy_from_slice(&cell_colour);
            }
        }
        println!("draw: {}", now.elapsed().unwrap().as_micros());
    }
}

// COPYPASTE: ideally this could be shared.

/// Create a window for the game.
///
/// Automatically scales the window to cover about 2/3 of the monitor height.
///
/// # Returns
///
/// Tuple of `(window, surface, width, height, hidpi_factor)`
/// `width` and `height` are in `PhysicalSize` units.
fn create_window(
    title: &str,
    event_loop: &EventLoop<()>,
) -> (winit::window::Window, pixels::wgpu::Surface, u32, u32, f64) {
    // Create a hidden window so we can estimate a good default window size
    let window = winit::window::WindowBuilder::new()
        .with_visible(false)
        .with_title(title)
        .build(&event_loop)
        .unwrap();
    let hidpi_factor = window.scale_factor();

    // Get dimensions
    let width = COLUMNS as f64;
    let height = ROWS as f64;
    let (monitor_width, monitor_height) = {
        let size = window.current_monitor().size();
        (
            size.width as f64 / hidpi_factor,
            size.height as f64 / hidpi_factor,
        )
    };
    let scale = (monitor_height / height * 2.0 / 3.0).round();

    // Resize, center, and display the window
    let min_size: winit::dpi::LogicalSize<f64> =
        PhysicalSize::new(width, height).to_logical(hidpi_factor);
    let default_size = LogicalSize::new(width * scale, height * scale);
    let center = LogicalPosition::new(
        (monitor_width - width * scale) / 2.0,
        (monitor_height - height * scale) / 2.0,
    );
    window.set_inner_size(default_size);
    window.set_min_inner_size(Some(min_size));
    window.set_outer_position(center);
    window.set_visible(true);

    let surface = pixels::wgpu::Surface::create(&window);
    let size = default_size.to_physical::<f64>(hidpi_factor);

    (
        window,
        surface,
        size.width.round() as u32,
        size.height.round() as u32,
        hidpi_factor,
    )
}