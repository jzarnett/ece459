#define EPS 1e-10
#define BIN_SIDE 100.0f

void bodyBodyInteraction(float4 bi, float4 bj, float4 * ai) {
    float4 r;
    
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    r.w = 1.0f;
    
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;
    
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = rsqrt(distSixth);
    
    float s = bj.w * invDistCube;
    
    (*ai).x += r.x * s;
    (*ai).y += r.y * s;
    (*ai).z += r.z * s;
    (*ai).w = 0.0f;
}

__kernel void addCMs(int points, __global float4* globalP, __global float4* globalCM) 
{
    int bin = get_global_id(0);
    int xbin = bin / 100;
    int ybin = bin % 100 / 10;
    int zbin = bin % 10;
    int i;

    globalCM[bin].x = globalCM[bin].y = globalCM[bin].z = 0.0f;
    globalCM[bin].w = 0.0f;
    for (i = 0; i < points; i++) {
        if ((int)(globalP[i].x/BIN_SIDE) == xbin && 
            (int)(globalP[i].y/BIN_SIDE) == ybin &&
            (int)(globalP[i].z/BIN_SIDE) == zbin) {
            globalCM[bin].x += globalP[i].x;
            globalCM[bin].y += globalP[i].y;
            globalCM[bin].z += globalP[i].z;
            globalCM[bin].w += 1.0f;
        }
    }

    globalCM[bin].x /= globalCM[bin].w;
    globalCM[bin].y /= globalCM[bin].w;
    globalCM[bin].z /= globalCM[bin].w;
}

__kernel void bin(int points, int binmax, __global float4* globalP, __global float4 *globalBins) {
    // global_id: in [0, 1000]; x * 100 + y * 10 + z
    int bin = get_global_id(0);
    int xbin = bin / 100;
    int ybin = bin % 100 / 10;
    int zbin = bin % 10;
    int bincount = 0;
    int i;

    for (i = 0; i < points; i++) {
	if ((int)(globalP[i].x/BIN_SIDE) == xbin && 
	    (int)(globalP[i].y/BIN_SIDE) == ybin &&
	    (int)(globalP[i].z/BIN_SIDE) == zbin) {
	    globalBins[bin*binmax+bincount++] = globalP[i];
	}
    }
}

#define toBin(x,y,z) (x*100+y*10+z)
#define BINMAX 10
inline int validBin(int x, int y, int z) {
  return (x >= 0) && (x < BINMAX) && (y >= 0) && (y < BINMAX) && (z >= 0) && (z < BINMAX);
}

// contribute acc for points in the current bin, and
// subtract out the CM force (which we will add)
void calculateExactForceForBin(int bin, int binmax, float4 myPosition, __global float4 * globalP, __global float4 * globalCM, __global float4 * globalBins, float4 * pacc) {
    float4 negBin;
    int i;
    for (i = 0; i < globalCM[bin].w; i ++) {
        bodyBodyInteraction(myPosition, globalBins[bin*binmax+i], pacc);
    }

    negBin.x = 2*myPosition.x-globalCM[bin].x;
    negBin.y = 2*myPosition.y-globalCM[bin].y;
    negBin.z = 2*myPosition.z-globalCM[bin].z;
    negBin.w = globalCM[bin].w;
    bodyBodyInteraction(myPosition, negBin, pacc);
}

// a work-item computes the forces for one point.
__kernel void calculateForces(int binmax, __global float4 * globalP, __global float4 * globalCM, __global float4 * globalBins, __global float4 * globalF) {
    int global_id = get_global_id(0);
    float4 myPosition = globalP[global_id];
    int xbin = (int)(globalP[global_id].x/BIN_SIDE),
      ybin = (int)(globalP[global_id].y/BIN_SIDE),
      zbin = (int)(globalP[global_id].z/BIN_SIDE);
    //int myBin = toBin(xbin, ybin, zbin);
    int i;
    float4 acc = {0.0f, 0.0f, 0.0f, 1.0f};
    int xp, yp, zp;

    // we have annoying corner cases; one way to get rid of them is to
    // just put a border (no points in extreme bins) around the bins
    // array, but we won't do that.

    // step 1: compute exact forces for nearby bins;
    // also subtract out the forces for these bins, so that we don't
    // float-count.
    for (xp = -1; xp <= 1; xp++) {
      for (yp = -1; yp <= 1; yp++) {
    	for (zp = -1; zp <= 1; zp++) {
    	  int xx, yy, zz;
    	  xx = xbin+xp; yy = ybin+yp; zz = zbin+zp;
    	  if (validBin(xx,yy,zz)) calculateExactForceForBin(toBin(xx,yy,zz), binmax, myPosition, globalP, globalCM, globalBins, &acc);
    	}
      }
    }

    // step 2: iterate among all bins and compute the forces for them.
    for (i = 0; i < 1000; i ++) {
      bodyBodyInteraction(myPosition, globalCM[0], &acc);
    }
    globalF[get_global_id(0)] = acc;
};

