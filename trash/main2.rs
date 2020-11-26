use ggez::{graphics, Context, ContextBuilder, GameResult};
use ggez::event::{self, EventHandler, KeyCode, KeyMods};
use ggez::nalgebra as na;
use na::{U2, U3, Dynamic, Matrix, MatrixArray, MatrixVec, Vector3};
use typenum::U200;
use std::collections::HashMap;

//type OtherMatrix = Matrix<f64, U200, U200, MatrixArray<f64, U200, U200>>;
type OtherMatrix = Matrix<f64, Dynamic, Dynamic, MatrixVec<f64, Dynamic, Dynamic>>;

const C: f64 = 1.0;
const W: f64 = 0.05;

fn main() {
    // Make a Context.
    //let mut contacts = HashMap::new();
    let a = OtherMatrix::new_random(200, 200);
    let b = OtherMatrix::new_random(200, 200);
    let c = OtherMatrix::new_random(200, 200);
    let d = OtherMatrix::new_random(200, 200);
    let e = OtherMatrix::new_random(200, 200);
    let f = OtherMatrix::new_random(200, 200);
    let g = OtherMatrix::new_random(200, 200);
    let h = OtherMatrix::new_random(200, 200);
    let i = OtherMatrix::new_random(200, 200);
    let j = OtherMatrix::new_random(200, 200);
    let k = OtherMatrix::new_random(200, 200);
    let l = OtherMatrix::new_random(200, 200);

    //let a = MyMatrix::new_random(200, 200);
    //let b = MyMatrix::new_random(200, 200);
    //let d = MyMatrix::new_random(200, 200);

    //let c = vec![a, b, d];

    //println!["{}", c[0][(199, 199)]];

    let veccy = vec![a, b, c, d, e, f, g, h, i, j, k, l];

    println!("{}", veccy.sum()[(199, 199)]);
    //contacts.insert("hi", MyMatrix::new_random(200, 200));
    //contacts.insert("bye", MyMatrix::new_random(200, 200));

    //let thing = OtherMatrix::new_random(200, 200);
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

struct MyGame {
    // Your state here...
    // densities starting at vector e0
    cells: Vec<Vec<Density>>,
    total_velocities: Vec<Vec<Vector2d>>,
    total_densities: Vec<Vec<f64>>,
    prob_vec: Vec<f64>,
    disp_vec: Vec<(f64, f64)>,
    count: usize,

    //density: Vec<Matrix200x200<f64>>,

}

#[derive(Clone, Debug)]
struct Density {
    eq_density: Vec<f64>,
    density: Vec<f64>,
}

impl Density {
    pub fn new() -> Density {
        Density {
            eq_density: vec![2.0; 9],
            density: vec![10.0; 9],
        }
    }
}

impl MyGame {
    pub fn new(_ctx: &mut Context) -> MyGame {
        // Load/create resources such as images here.
        let len_column = 51;
        let len_row = 51;
        let disp_vec: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (-1.0, 0.0), (0.0, -1.0), (1.0, 1.0), 
        (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];

        let prob_vec: Vec<f64> = vec![4.0/9.0, 1.0/9.0, 1.0/9.0, 
        1.0/9.0, 1.0/9.0, 1.0/36.0, 
        1.0/36.0, 1.0/36.0, 1.0/36.0];

        MyGame {
            cells: vec![vec![Density::new(); len_column]; len_row],
            total_velocities: vec![vec![Vector2d::new(); len_column]; len_row],
            total_densities: vec![vec![0.0; len_column]; len_row],
            prob_vec: prob_vec,
            disp_vec: disp_vec,
            count: 0,
        }
    }

    pub fn cell_params(&mut self) {
        let len_row = self.cells.len();
        let len_column = self.cells[0].len();

        for m in 0..len_row {
            for n in 0..len_column {
                let density = &self.cells[m][n].density;
                self.total_densities[m][n] = density.into_iter().sum();

                let mut total_velocity_x = 2.0;
                let mut total_velocity_y = 0.0;

                total_velocity_x = (density[1] + density[2] - density[4] - density[5] - density[6] + density[8]) / self.total_densities[m][n];
                total_velocity_y = (density[2] + density[3] + density[4] - density[6] - density[7] - density[8]) / self.total_densities[m][n];
                

                self.total_velocities[m][n] = Vector2d {
                    x: total_velocity_x,
                    y: total_velocity_y,
                };


                //squared total velocity
                //total_velocities[m].push(total_velocity);
                //velocities[i].push(density.into_iter().map(|x| (e * c).powi(2) *  ).collect());
            }
        }

    }

    pub fn new_density(&mut self) {
        let len_row = self.cells.len();
        let len_column = self.cells[0].len();

        let mut new_densities = vec![vec![Density::new(); len_column]; len_row];

        for m in 0..len_row {
            for n in 0..len_column {
                let total_density = self.total_densities[m][n];
                let total_velocity = self.total_velocities[m][n];

                
                for i in 0..9 {

                    let disp_vec_i = Vector2d {
                        x: self.disp_vec[i].0,
                        y: self.disp_vec[i].1,
                    };
                    let dot_prod = disp_vec_i.dot(total_velocity);
                    let n_eq = total_density * self.prob_vec[i] * (1.0 + 3.0 * dot_prod
                    + 9.0 / 2.0 * dot_prod.powi(2) - 3.0 / 2.0 * total_velocity.sq_hypot());

                    let n_old = self.cells[m][n].density[i];

                    let n_new = n_old + W * (n_eq - n_old);
                    //println!("yo {}", n_new);

                    new_densities[m][n].density[i] = n_new;
                    new_densities[m][n].eq_density[i] = n_eq;

                }
            }
        }
        
        self.cells = new_densities;
    }

    pub fn stream(&mut self) {
        let len_row = self.cells.len();
        let len_column = self.cells[0].len();

        

        let mut new_cells = vec![vec![Density::new(); len_column]; len_row];

        for m in 0..len_row {
            for n in 0..len_column {
                //println!("yo {} {}", m, n);
                let density = self.cells[m][n].clone();
                
                match (m, n) {
                    //corners
                    (0, 0) => new_cells[m][n].density = density.eq_density,
                    (0, 50) => new_cells[m][n].density = density.eq_density,
                    (50, 50) => {
                        
                        new_cells[m][n].density = density.eq_density;
                    },
                    (50, 0) => new_cells[m][n].density = density.eq_density,
                    //sides
                    (0, _) => new_cells[m][n].density = density.eq_density,
                    (_, 50) => new_cells[m][n].density = density.eq_density,
                    (50, _) => new_cells[m][n].density = density.eq_density,
                    (_, 0) => new_cells[m][n].density = density.eq_density,
                    (_, _) => for i in 0..9 {
                        match i {
                            0 => new_cells[m][n].density[i] = density.density[i],
                            1 => new_cells[m][n + 1].density[i] = density.density[i],
                            2 => new_cells[m - 1][n].density[i] = density.density[i],
                            3 => new_cells[m][n - 1].density[i] = density.density[i],
                            4 => new_cells[m + 1][n].density[i] = density.density[i],
                            5 => new_cells[m - 1][n + 1].density[i] = density.density[i],
                            6 => new_cells[m - 1][n - 1].density[i] = density.density[i],
                            7 => new_cells[m + 1][n - 1].density[i] = density.density[i],
                            8 => new_cells[m + 1][n + 1].density[i] = density.density[i],
                            _ => (),
                        }
                    },
                }
                
            }
        }
        self.cells = new_cells;
    }
}

impl EventHandler for MyGame {
    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        // Update code here...
        self.cell_params();
        self.new_density();
        self.stream();
        self.count += 1;

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, graphics::WHITE);
        // Draw code here...
        
        let len_row = self.cells.len();
        let len_column = self.cells[0].len();
        let width = 5.0;

        let v_max = 0.4;

        for m in 0..len_row {
            for n in 0..len_column {
                
                let velocity = (self.total_velocities[m][n].x.powi(2) + self.total_velocities[m][n].y.powi(2)).sqrt();
                let vel_colour: [f32; 4] = [1.0, 0.0, 0.0, (velocity / v_max) as f32];


                let rect = graphics::Mesh::new_rectangle(
                    ctx,
                    graphics::DrawMode::fill(),
                    graphics::Rect::new(0.0, 0.0, width + 1.0, width + 1.0),
                    vel_colour.into(),
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
        graphics::present(ctx)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vector2d {
    x: f64,
    y: f64,
}

impl Vector2d {
    pub fn new() -> Vector2d {
        Vector2d { x: 0.0, y: 0.0 }
    }

    pub fn sum(&self) -> f64 {
        self.x + self.y
    }

    pub fn subtract(&self, other: &Vector2d) -> Vector2d {
        Vector2d {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn add(&self, other: &Vector2d) -> Vector2d {
        Vector2d {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn multiply(&self, multiplier: f64) -> Vector2d {
        Vector2d {
            x: self.x * multiplier,
            y: self.y * multiplier,
        }
    }

    pub fn divide(&self, divisor: f64) -> Vector2d {
        Vector2d {
            x: self.x / divisor,
            y: self.y / divisor,
        }
    }

    pub fn dot(&self, other: Vector2d) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn root(&self) -> Vector2d {
        Vector2d {
            x: self.x.abs().sqrt(),
            y: self.y.abs().sqrt(),
        }
    }

    pub fn sq_hypot(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2)
    }
}