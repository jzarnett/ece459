kernel void contains_branch(global float *a, 
                            global float *b) {
    int id = get_global_id(0);
    if (cond) {
        x[id] += 5.0;
    } else {
        y[id] += 5.0;
    }
}
