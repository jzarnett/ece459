#define EPS 1e-10

float4 bodyBodyInteraction(float4 bi, float4 bj, float4 ai) {
    float4 r;
    
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    r.w = 1.0f;
    
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;
    
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = rsqrt(distSixth);
    
    float s = bj.w * invDistCube;
    
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    
    return ai;
}

// a work-item computes the forces for one point.
__kernel void calculateForces(__global float4 * globalP, __global float4 * globalF) {
    float4 myPosition;
    uint i;
    
    float4 acc = {0.0f, 0.0f, 0.0f, 1.0f};
    
    myPosition = globalP[get_global_id(0)];
    
    for (i = 0; i < get_global_size(0); i++) {
	acc = bodyBodyInteraction(myPosition, globalP[i], acc);
    }
    globalF[get_global_id(0)] = acc;
};

