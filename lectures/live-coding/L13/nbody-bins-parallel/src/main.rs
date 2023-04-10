use rand::Rng;
use std::fmt::{Display, Formatter};
use rayon::prelude::*;

/* A Rustification by Jeff Zarnett of a past ECE 459 N-Body assignment that was
originally from GPU Gems, Chapter 31 and modified by Patrick Lam. */

const NUM_POINTS: usize = 100000;
const EPSILON: f32 = 1e-10;
const SPACE: f32 = 1000.0;
const BIN_SIDE: i32 = 100;
const NUM_BINS: usize = 1000;

struct Point {
    x: f32,
    y: f32,
    z: f32,
    mass: f32,
    bin: usize,
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

impl Display for Acceleration {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

fn body_body_interaction(
    current_point: &Point, other_point: &Point, current_accel: &mut Acceleration,
) {
    let difference = Point {
        x: current_point.x - other_point.x,
        y: current_point.y - other_point.y,
        z: current_point.z - other_point.z,
        mass: 1.0,
        bin: 0,
    };
    let distance_squared = difference.x * difference.x +
        difference.y * difference.y +
        difference.z * difference.z +
        EPSILON;
    let distance_sixth = distance_squared * distance_squared * distance_squared;
    let inv_dist_cube = 1.0f32 / (distance_sixth).sqrt();
    let magnitude = other_point.mass * inv_dist_cube;

    current_accel.x = current_accel.x + difference.x * magnitude;
    current_accel.y = current_accel.y + difference.y * magnitude;
    current_accel.z = current_accel.z + difference.z * magnitude;
}

fn calculate_forces(initial_positions: Vec<Point>, centres_of_mass: Vec<Point>) -> Vec<Acceleration> {
    let mut accelerations: Vec<Acceleration> = initialize_accelerations();
    accelerations.par_iter_mut().enumerate().for_each(|(i, current_accel)| {
        let current_point: &Point = initial_positions.get(i).unwrap();
        for j in 0..NUM_POINTS {
            let other_point: &Point = initial_positions.get(j).unwrap();
            if is_adjacent(current_point.bin as i32, other_point.bin as i32) {
                body_body_interaction(current_point, other_point, current_accel);
            }
        }
        for k in 0..NUM_BINS {
            let centre = centres_of_mass.get(k).unwrap();
            if !is_adjacent(current_point.bin as i32, centre.bin as i32) {
                body_body_interaction(current_point, centre, current_accel);
            }
        }
    });
    return accelerations;
}

fn initialize_positions_and_centres_of_mass() -> (Vec<Point>, Vec<Point>) {
    let mut initial_positions: Vec<Point> = Vec::new();
    let mut centres_of_mass: Vec<Point> = Vec::new();
    let mut rng = rand::thread_rng();

    for j in 0..NUM_BINS {
        centres_of_mass.push(Point {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            mass: 0.0,
            bin: j,
        })
    }

    for _i in 0..NUM_POINTS {
        let point_x = rng.gen_range(0.0, SPACE);
        let point_y = rng.gen_range(0.0, SPACE);
        let point_z = rng.gen_range(0.0, SPACE);
        let point_mass = rng.gen_range(0.01, 100.0);

        let bin = calculate_bin(point_x, point_y, point_z);

        initial_positions.push(Point {
            x: point_x,
            y: point_y,
            z: point_z,
            mass: point_mass,
            bin: bin,
        });
        let bin_of_this_point = centres_of_mass.get_mut(bin).unwrap();
        bin_of_this_point.x += point_mass * point_x;
        bin_of_this_point.y += point_mass * point_y;
        bin_of_this_point.z += point_mass * point_z;
        bin_of_this_point.mass += point_mass;
    }

    for k in 0..NUM_BINS {
        let com: &mut Point = centres_of_mass.get_mut(k).unwrap();
        if com.mass != 0f32 {
            com.x /= com.mass;
            com.y /= com.mass;
            com.z /= com.mass;
        }
    }

    (initial_positions, centres_of_mass)
}

fn calculate_bin(x: f32, y: f32, z: f32) -> usize {
    let x_bin = (x as i32) / BIN_SIDE;
    let y_bin = (y as i32) / BIN_SIDE;
    let z_bin = (z as i32) / BIN_SIDE;

    (x_bin * 100 + y_bin * 10 + z_bin) as usize
}

fn initialize_accelerations() -> Vec<Acceleration> {
    let mut result: Vec<Acceleration> = Vec::new();
    for _i in 0..NUM_POINTS {
        result.push(Acceleration {
            x: 0f32,
            y: 0f32,
            z: 0f32,
        })
    }
    result
}

fn is_adjacent(bin1: i32, bin2: i32) -> bool {
    //let ADJACENCY_VECTOR: [i32; 27] = [00, 01, 11, 10, 9, -01, -11, -10, -9, 100, 101, 111, 110, 109, -101, -111, -110, -109, -100, -99, -89, -90, -91, 99, 89, 90, 91];
    // Okay, so you're wondering what's up with this horrifying if-condition with all the OR
    // operators. I don't blame you! Originally, I wrote this to use the ADJACENCY_VECTOR that is
    // commented out above, and iterate over that array and add the current index to bin2 and then
    // return true. This was much, much slower than the unmodified program. I found it out by
    // running this program with the profiler, and the profiler told me that lots of time was being
    // spent in the code that generates the slice for this comparison.
    if bin1 == bin2 ||
        bin1 == (bin2 + 1) ||
        bin1 == (bin2 + 11) ||
        bin1 == (bin2 + 10) ||
        bin1 == (bin2 + 9) ||
        bin1 == (bin2 + -1) ||
        bin1 == (bin2 + -11) ||
        bin1 == (bin2 + -10) ||
        bin1 == (bin2 + -9) ||
        bin1 == (bin2 + 100) ||
        bin1 == (bin2 + 101) ||
        bin1 == (bin2 + 111) ||
        bin1 == (bin2 + 110) ||
        bin1 == (bin2 + 109) ||
        bin1 == (bin2 + -101) ||
        bin1 == (bin2 + -111) ||
        bin1 == (bin2 + -110) ||
        bin1 == (bin2 + -109) ||
        bin1 == (bin2 + -100) ||
        bin1 == (bin2 + -99) ||
        bin1 == (bin2 + -89) ||
        bin1 == (bin2 + -90) ||
        bin1 == (bin2 + -91) ||
        bin1 == (bin2 + 99) ||
        bin1 == (bin2 + 89) ||
        bin1 == (bin2 + 90) ||
        bin1 == (bin2 + 91)
    {
        return true;
    }
    false
}

fn main() {
    let (initial_positions, centres_of_mass) = initialize_positions_and_centres_of_mass();
    println! {"Initial positions:"}
    for pt in initial_positions.iter() {
        println! {"{}", pt};
    }

    let final_accelerations = calculate_forces(initial_positions, centres_of_mass);
    println! {"Accelerations:"}
    for accel in final_accelerations.iter() {
        println! {"{}", accel};
    }
}
