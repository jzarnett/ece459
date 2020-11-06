kernel void contains_loop(global float *a, 
                          global float *b) {
    int id = get_global_id(0);
    
    for (i = 0; i < id; i++) {
        b[i] += a[i];
    }
}
