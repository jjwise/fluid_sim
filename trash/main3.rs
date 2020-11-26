use ggez::{graphics, Context, ContextBuilder, GameResult};
use ggez::event::{self, EventHandler, KeyCode, KeyMods};
use ggez::nalgebra as na;
use na::{U2, U3, Dynamic, Matrix, MatrixArray, MatrixVec, Vector2};
use typenum::U200;
use std::time::{Duration, SystemTime};
use std::thread::sleep;


//type MyMatrix = Matrix<f64, U200, U200, MatrixArray<f64, U200, U200>>;
type MyMatrix = Matrix<f64, Dynamic, Dynamic, MatrixVec<f64, Dynamic, Dynamic>>;
type BoolMatrix = Matrix<bool, Dynamic, Dynamic, MatrixVec<bool, Dynamic, Dynamic>>;

const C: f64 = 1.0;
const W: f64 = 1.0;
const ROWS: usize = 100;
const COLUMNS: usize = 100;

fn main() {
    // Make a Context.
    
    let (mut ctx, mut event_loop) = ContextBuilder::new("my_game", "Cool Game Author")
		.build()
		.expect("aieee, could not create ggez context!");

    // Create an instance of your event handler.
    // Usually, you should provide it with the Context object to
    // use when setting your game up.
    let mut my_game = MyGame::new(&mut ctx);

    // Run!
    match event::run(&mut ctx, &mut event_loop, &mut my_game) {
        Ok(_) => println!("Exited cleanly."),
        Err(e) => println!("Error occured: {}", e)
    }
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
    fn new() -> TotalVelocity {
        TotalVelocity {
            x: MyMatrix::repeat(ROWS, COLUMNS, 0.7),
            y: MyMatrix::repeat(ROWS, COLUMNS, 0.0),
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

struct MyGame {
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
}

impl MyGame {
    pub fn new(_ctx: &mut Context) -> MyGame {
        // Load/create resources such as images here.
        MyGame {
            // Your state here...
            // densities starting at vector e0
            rows: ROWS,
            columns: COLUMNS,
            norm_density: MyGame::initialise_density(),
            eq_density: MyGame::density(ROWS, COLUMNS),
            obstacles: MyGame::init_obstacles(),
            total_velocity: TotalVelocity::new(),
            total_density: MyMatrix::zeros(ROWS, COLUMNS),
            cell_consts: CellConsts::get_consts(),
            colour_map: ColourMap::gen_jet_colourmap(),
            count: 0,
        }
    }  

    pub fn init_obstacles() -> MyMatrix {
        let mut new_obstacles = MyMatrix::zeros(ROWS, COLUMNS);
        new_obstacles[(10, 10)] = 1.0;
        new_obstacles[(11, 10)] = 1.0;
        new_obstacles[(12, 10)] = 1.0;
        new_obstacles[(13, 10)] = 1.0;
        new_obstacles[(14, 10)] = 1.0;
        new_obstacles[(15, 10)] = 1.0;
        new_obstacles
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

        self.total_velocity.x = (&self.norm_density[1] +
        &self.norm_density[3] -
        &self.norm_density[5] -
        &self.norm_density[6] - 
        &self.norm_density[7] +
        &self.norm_density[8]).component_div(&self.total_density);
        //println!("{:?}", self.total_velocity.x);

        self.total_velocity.y = (&self.norm_density[2] +
        &self.norm_density[4] +
        &self.norm_density[5] -
        &self.norm_density[6] -
        &self.norm_density[7] -
        &self.norm_density[8]).component_div(&self.total_density);
    }

    pub fn initialise_density() -> Vec<MyMatrix> {
        let total_velocity = TotalVelocity::new();
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
        let pre_calculated = - 3.0 / 2.0 * self.total_velocity.sq_hypot();
        let ones = MyMatrix::repeat(ROWS, COLUMNS, 1.0);
        //println!("{:?}", pre_calculated);

        for i in 0..9 {
            let dot_prod = &self.total_velocity.x * self.cell_consts[i].disp[0] + &self.total_velocity.y * self.cell_consts[i].disp[1];
            let dot_prod_square = &dot_prod.component_mul(&dot_prod);
            self.eq_density[i] = (&self.total_density * self.cell_consts[i].prob).component_mul(&((&ones + 3.0 * dot_prod) + 
            4.5 * dot_prod_square - &pre_calculated));
            self.norm_density[i] = &self.norm_density[i] + W * (&self.eq_density[i] - &self.norm_density[i]);
        }
    }

    pub fn stream(&mut self) {
        unsafe {
            let mut norm_density_e = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_e.copy_from(&self.norm_density[1]);

            self.norm_density[1] = norm_density_e.remove_column(COLUMNS - 1).insert_column(0, 1.0 / 9.0); //e

            let mut norm_density_ne = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_ne.copy_from(&self.norm_density[5]);

            self.norm_density[5] = norm_density_ne.remove_row(0).remove_column(COLUMNS - 1).insert_row(ROWS - 1, 4.0 / 36.0) //ne 5
            .insert_column(0, 4.0 / 36.0);

            let mut norm_density_n = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_n.copy_from(&self.norm_density[2]);

            self.norm_density[2] = norm_density_n.remove_row(0).insert_row(ROWS - 1, 1.0 / 9.0); //n

            let mut norm_density_nw = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_nw.copy_from(&self.norm_density[6]);

            self.norm_density[6] = norm_density_nw.remove_row(0).remove_column(0).insert_row(ROWS - 1, 4.0 / 36.0)//nw 6
            .insert_column(COLUMNS - 1, 4.0 / 36.0);

            let mut norm_density_w = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_w.copy_from(&self.norm_density[3]);

            self.norm_density[3] = norm_density_w.remove_column(0).insert_column(COLUMNS - 1, 1.0 / 9.0); //w

            let mut norm_density_sw = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_sw.copy_from(&self.norm_density[7]);

            self.norm_density[7] = norm_density_sw.remove_row(ROWS - 1).remove_column(0).insert_row(0, 4.0 / 36.0) //sw 7
            .insert_column(COLUMNS - 1, 4.0 / 36.0);

            let mut norm_density_s = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_s.copy_from(&self.norm_density[4]);

            self.norm_density[4] = norm_density_s.remove_row(ROWS - 1).insert_row(0, 1.0 / 9.0); //s

            let mut norm_density_se = MyMatrix::new_uninitialized(ROWS, COLUMNS);
            norm_density_se.copy_from(&self.norm_density[8]);

            self.norm_density[8] = norm_density_se.remove_row(ROWS - 1).remove_column(COLUMNS - 1).insert_row(0, 4.0 / 36.0) //se 8
            .insert_column(0, 4.0 / 36.0);
        }

        //println!("{:?}", self.norm_density[8]);
        for m in 0..ROWS {
            for n in 0..COLUMNS {
                if self.obstacles[(m, n)] == 1.0 {
                    for i in 0..8 {
                        let coords = ((m as f64 + self.cell_consts[i + 1].disp[1]) as usize, (n as f64 + self.cell_consts[i + 1].disp[0]) as usize);
                        let m = m as usize;
                        let n = n as usize;
                        let cell = self.obstacles[(m, n)];
                        if self.obstacles[coords] == 0.0 {
                            //println!("hi");
                            match i {
                                0 => {
                                    self.norm_density[1][coords] = self.norm_density[3][(m, n)];
                                },
                                1 => {
                                    self.norm_density[2][coords] = self.norm_density[4][(m, n)];
                                },
                                2 => {
                                    self.norm_density[3][coords] = self.norm_density[1][(m, n)];
                                },
                                3 => {
                                    self.norm_density[4][coords] = self.norm_density[2][(m, n)];
                                },
                                4 => {
                                    self.norm_density[5][coords] = self.norm_density[7][(m, n)];
                                },
                                5 => {
                                    self.norm_density[6][coords] = self.norm_density[8][(m, n)];
                                },
                                6 => {
                                    self.norm_density[7][coords] = self.norm_density[5][(m, n)];
                                },
                                7 => {
                                    self.norm_density[8][coords] = self.norm_density[6][(m, n)];
                                },
                                _ => (),
                            }
                        }
                    }
                }
            }
        }
    }
}

impl EventHandler for MyGame {
    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        // Update code here...
        let now = SystemTime::now();
        self.update_cell_macro();
        println!("update_cell_macro(): {}", now.elapsed().unwrap().as_micros());
        let now = SystemTime::now();
        self.new_density();
        println!("new_density(): {}", now.elapsed().unwrap().as_micros());
        let now = SystemTime::now();
        self.stream();
        println!("stream: {}", now.elapsed().unwrap().as_micros());
        self.count += 1;
        

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        let now = SystemTime::now();
        graphics::clear(ctx, graphics::WHITE);
        // Draw code here...
        
        let width = 5.0;

        let v_max = 0.4;

        for m in 0..ROWS {
            for n in 0..COLUMNS {
                
                let velocity = (self.total_velocity.x[(m, n)].powi(2) + self.total_velocity.y[(m, n)].powi(2)).sqrt();
                let mut index = (velocity * 4.0 * 400.0) as usize;

                if index > 399 {
                    index = 399;
                } else if index < 0 {
                    index = 0;
                };

                let r = self.colour_map.red_list[index] as f32 / 255.0;
                let g = self.colour_map.green_list[index] as f32 / 255.0;
                let b = self.colour_map.blue_list[index] as f32 / 255.0;

                let cell_colour: [f32; 4];

                if self.obstacles[(m, n)] == 1.0 {
                    cell_colour = [0.0, 0.0, 0.0, 1.0];
                } else {
                    cell_colour = [r, g, b, 1.0];
                }

                let rect = graphics::Mesh::new_rectangle(
                    ctx,
                    graphics::DrawMode::fill(),
                    graphics::Rect::new(0.0, 0.0, width + 1.0, width + 1.0),
                    cell_colour.into(),
                ).unwrap();

                graphics::draw(
                    ctx,
                    &rect,
                    (na::Point2::new(
                        n as f32 * width - 1.0,
                        m as f32 * width - 1.0,
                    ),),
                ).unwrap();

            }
        }
        println!("draw: {}", now.elapsed().unwrap().as_micros());
        graphics::present(ctx)
    }
}
