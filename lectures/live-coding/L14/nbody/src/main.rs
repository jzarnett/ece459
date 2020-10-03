use rand::Rng;
use std::fmt::{Display, Formatter};

/* A Rustification by Jeff Zarnett of a past ECE 459 N-Body assignment that was
originally from GPU Gems, Chapter 31 and modified by Patrick Lam. */

const NUM_POINTS: u32 = 5000;
const EPSILON: f32 = 1e-10;
const SPACE: f32 = 1000.0;

struct Point {
    x: f32,
    y: f32,
    z: f32,
    mass: f32,
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}) [{}]", self.x, self.y, self.z, self.mass)
    }
}

struct Acceleration {
    x: f32,
    y: f32,
    z: f32,
}

impl Display for Acceleration{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

fn body_body_interaction(
    current_point: &Point, other_point: &Point, current_accel: &mut Acceleration
) {
    let difference = Point {
        x: current_point.x - other_point.x,
        y: current_point.y - other_point.y,
        z: current_point.z - other_point.z,
        mass: 1.0,
    };
    let distance_squared = difference.x * difference.x +
        difference.y * difference.y +
        difference.z * difference.z +
        EPSILON;
    let distance_sixth = distance_squared * distance_squared * distance_squared;
    let inv_dist_cube = 1.0f32 / distance_sixth;
    let magnitude = current_point.mass * inv_dist_cube;

    current_accel.x = current_accel.x + difference.x * magnitude;
    current_accel.y = current_accel.y + difference.y * magnitude;
    current_accel.z = current_accel.z + difference.z * magnitude;
}

fn calculate_forces(initial_positions: Vec<Point>) -> Vec<Acceleration> {
    let mut accelerations: Vec<Acceleration> = initialize_accelerations();
    for i in 0 .. NUM_POINTS {
        let current_point: &Point = initial_positions.get(i as usize).unwrap();
        let current_accel: &mut Acceleration = accelerations.get_mut(i as usize).unwrap();
        for j in 0 .. NUM_POINTS {
            let other_point: &Point = initial_positions.get(j as usize).unwrap();
            body_body_interaction(current_point, other_point, current_accel);
        }
    }

    return accelerations;
}

fn initialize_positions() -> Vec<Point> {
    let mut result: Vec<Point> = Vec::new();
    let mut rng = rand::thread_rng();

    for _i in 0..NUM_POINTS {
        result.push(Point {
            x: rng.gen_range(-1.0f32 * SPACE, SPACE),
            y: rng.gen_range(-1.0f32 * SPACE, SPACE),
            z: rng.gen_range(-1.0f32 * SPACE, SPACE),
            mass: rng.gen_range(0.01, 100.0),
        });
    }
    result
}

fn initialize_accelerations() -> Vec<Acceleration> {
    let mut result: Vec<Acceleration> = Vec::new();
    for _i in 0 .. NUM_POINTS {
        result.push(Acceleration {
            x: 0f32,
            y: 0f32,
            z: 0f32,
        })
    }
    result
}


fn main() {
    let initial_positions = initialize_positions();
    println! {"Initial positions:"}
    for pt in initial_positions.iter() {
        println! {"{}", pt};
    }

    let final_accelerations = calculate_forces(initial_positions);
    println! {"Accelerations:"}
    for accel in final_accelerations.iter() {
        println! {"{}", accel};
    }
}
