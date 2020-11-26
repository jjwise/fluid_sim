use ggez::{graphics, Context, ContextBuilder, GameResult};
use ggez::event::{self, EventHandler, KeyCode, KeyMods};
use ggez::nalgebra as na;
use na::{U2, U3, Dynamic, Matrix, MatrixArray, MatrixVec, Vector2};
use typenum::U200;


//type MyMatrix = Matrix<f64, U200, U200, MatrixArray<f64, U200, U200>>;
type MyMatrix = Matrix<Vector2, U200, U200, MatrixArray<f64, U200, U200>>;

const C: f64 = 1.0;
const W: f64 = 0.2;

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

#[derive(Clone, Debug)]
struct TotalVelocity {
    x: MyMatrix,
    y: MyMatrix,
}

impl TotalVelocity {
    fn new() -> TotalVelocity {
        x: MyMatrix::zeros(),
        y: MyMatrix::zeros(),
    }
}

#[derive(Clone, Debug)]
struct Density {
    mid: MyMatrix,
    e: MyMatrix,
    n: MyMatrix,
    w: MyMatrix,
    s: MyMatrix,
    ne: MyMatrix,
    nw: MyMatrix,
    sw: MyMatrix,
    se: MyMatrix,
}

impl Density {
    pub fn new() -> Density {
        Density {
            mid: MyMatrix::zeros(),
            e: MyMatrix::zeros(),
            n: MyMatrix::zeros(),
            w: MyMatrix::zeros(),
            s: MyMatrix::zeros(),
            ne: MyMatrix::zeros(),
            nw: MyMatrix::zeros(),
            sw: MyMatrix::zeros(),
            se: MyMatrix::zeros(),
        }
    }

    pub fn sum(&self) -> MyMatrix {
        self.mid +
        self.e +
        self.n +
        self.w +
        self.s +
        self.ne +
        self.nw +
        self.sw +
        self.se
    }
}

struct CellConsts {
    mid: Vec<Consts>,
    e: Vec<Consts>,
    n: Vec<Consts>,
    w: Vec<Consts>,
    s: Vec<Consts>,
    ne: Vec<Consts>,
    nw: Vec<Consts>,
    sw: Vec<Consts>,
    se: Vec<Consts>,
}

impl CellConsts {
    fn get_consts() -> CellConsts {
        CellConsts {
            mid: Consts {
                prob: 4.0 / 9.0,
                disp: Vector2::new(0.0, 0.0),
            },
            e: Consts {
                prob: 1.0 / 9.0,
                disp: Vector2::new(1.0, 0.0),
            },
            n: Consts {
                prob: 1.0 / 9.0,
                disp: Vector2::new(0.0, 1.0),
            },
            w: Consts {
                prob: 1.0 / 9.0,
                disp: Vector2::new(-1.0, 0.0),
            },
            s: Consts {
                prob: 1.0 / 9.0,
                disp: Vector2::new(0.0, -1.0),
            },
            ne: Consts {
                prob: 1.0 / 36.0,
                disp: Vector2::new(1.0, 1.0),
            },
            nw: Consts {
                prob: 1.0 / 36.0,
                disp: Vector2::new(-1.0, 1.0),
            },
            sw: Consts {
                prob: 1.0 / 36.0,
                disp: Vector2::new(-1.0, -1.0),
            },
            se: Consts {
                prob: 1.0 / 36.0,
                disp: Vector2::new(1.0, -1.0),
            },
        }
    }
}

struct Consts {
    prob: f64,
    disp: Vector2<f64>,
}

struct MyGame {
    height: usize,
    width: usize,
    norm_density: Density,
    eq_density: Density,
    total_velocity: TotalVelocity,
    total_density: MyMatrix,
    cell_consts: CellConsts,
    count: usize,
}

impl MyGame {
    pub fn new(_ctx: &mut Context) -> MyGame {
        // Load/create resources such as images here.
        MyGame {
            // Your state here...
            // densities starting at vector e0
            rows: 200,
            columns: 200,
            norm_density: Density::new(),
            eq_density: Density::new(),
            total_velocity: TotalVelocity::new(),
            total_density: MyMatrix::zeros(),
            cell_consts: CellConsts::get_consts(),
            count: 0,
        }
    }  

    pub fn update_cell_macro(&mut self) {
        self.total_density = self.norm_density.sum();

        self.total_velocity.x = self.norm_density.e +
        self.norm_density.ne -
        self.norm_density.nw -
        self.norm_density.w - 
        self.norm_density.sw +
        self.norm_density.se / self.total_density;

        self.total_velocity.y = (self.norm_density.ne +
        self.norm_density.n +
        self.norm_density.nw -
        self.norm_density.sw -
        self.norm_density.s -
        self.norm_density.se) / self.total_density;
    }

    pub fn collision_calc(&self, cell_disp: &Vec<f64>, cell_prob: f64) -> MyMatrix {

        let dot_prod = self.total_velocity.x * cell_disp[0] + self.total_velocity.y * cell_disp[1];

        self.total_density * self.cell_prob * (MyMatrix::ones() + 3.0 * dot_prod) + 9.0 / 2.0 * dot_prod.powi(2)
    }

    pub fn new_density(&mut self) {
        let pre_calculated = - 3.0 / 2.0 * total_velocity.sq_hypot();

        self.eq_density.mid = self.collision_calc(&self.cell_consts.mid.disp, self.cell_consts.mid.prob) - pre_calculated;
        self.norm_density.mid = norm_density.mid + W * (eq_density.mid - norm_density.mid)

        self.eq_density.e = self.collision_calc(&self.cell_consts.e.disp, self.cell_consts.e.prob) - pre_calculated;
        self.norm_density.e = norm_density.e + W * (eq_density.e - norm_density.e)

        self.eq_density.ne = self.collision_calc(&self.cell_consts.ne.disp, self.cell_consts.ne.prob) - pre_calculated;
        self.norm_density.ne = norm_density.ne + W * (eq_density.ne - norm_density.ne)

        self.eq_density.n = self.collision_calc(&self.cell_consts.n.disp, self.cell_consts.n.prob) - pre_calculated;
        self.norm_density.n = norm_density.n + W * (eq_density.n - norm_density.n)

        self.eq_density.nw = self.collision_calc(&self.cell_consts.nw.disp, self.cell_consts.nw.prob) - pre_calculated;
        self.norm_density.ne = norm_density.nw + W * (eq_density.nw - norm_density.nw)

        self.eq_density.w = self.collision_calc(&self.cell_consts.w.disp, self.cell_consts.w.prob) - pre_calculated;
        self.norm_density.w = norm_density.w + W * (eq_density.w - norm_density.w)

        self.eq_density.sw = self.collision_calc(&self.cell_consts.sw.disp, self.cell_consts.sw.prob) - pre_calculated;
        self.norm_density.sw = norm_density.sw + W * (eq_density.sw - norm_density.sw)

        self.eq_density.s = self.collision_calc(&self.cell_consts.s.disp, self.cell_consts.s.prob) - pre_calculated;
        self.norm_density.s = norm_density.s + W * (eq_density.s - norm_density.s)

        self.eq_density.se = self.collision_calc(&self.cell_consts.se.disp, self.cell_consts.se.prob) - pre_calculated;
        self.norm_density.se = norm_density.se + W * (eq_density.se - norm_density.se)
    }

    pub fn stream(&mut self) {
        let mut new_norm_density = MyMatrix::zeros();

        for m in 0..self.rows {
            for n in 0..self.columns {
                //println!("yo {} {}", m, n);

                match (m, n) {
                    //corners
                    (0, 0) => new_norm_density[m][n] = self.eq_density,
                    (0, 199) => new_norm_density[m][n] = self.eq_density,
                    (199, 199) => new_norm_density[m][n] = self.eq_density,
                    (199, 0) => new_norm_density[m][n] = self.eq_density,
                    //sides
                    (0, _) => new_norm_density[m][n] = self.eq_density,
                    (_, 199) => new_norm_density[m][n] = self.eq_density,
                    (199, _) => new_norm_density[m][n] = self.eq_density,
                    (_, 0) => new_norm_density[m][n] = self.eq_density,
                    (_, _) => for i in 0..9 {
                        match i {
                            0 => new_norm_density[m][n].density[i] = density.density[i],
                            1 => new_norm_density[m][n + 1].density[i] = density.density[i],
                            2 => new_norm_density[m - 1][n].density[i] = density.density[i],
                            3 => new_norm_density[m][n - 1].density[i] = density.density[i],
                            4 => new_norm_density[m + 1][n].density[i] = density.density[i],
                            5 => new_norm_density[m - 1][n + 1].density[i] = density.density[i],
                            6 => new_norm_density[m - 1][n - 1].density[i] = density.density[i],
                            7 => new_norm_density[m + 1][n - 1].density[i] = density.density[i],
                            8 => new_norm_density[m + 1][n + 1].density[i] = density.density[i],
                            _ => (),
                        }
                    },
                }
                
            }
        }
        self.cells = new_norm_density;
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
                
                let velocity = (self.total_velocities[m][n].x + self.total_velocities[m][n].y).sqrt();
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