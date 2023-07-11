/* Kernels for copying pixels */

/* #pragma OPENCL EXTENSION cl_khr_fp16 : enable */

// const sampler_t sampler =
//     CLK_NORMALIZED_COORDS_FALSE
//     | CLK_ADDRESS_CLAMP_TO_EDGE
//     | CLK_FILTER_NEAREST;

// __read_only image2d_t inp,
// float4 px = read_imagef(inp, sampler, (int2)(px_x, px_y));

__kernel void encode(__global uchar *restrict inp, uint img_w,
                     __global uchar *restrict out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    out[img_w * gid.y + gid.x] = inp[img_w * gid.y + gid.x];
}

__kernel void encode_y(__global uchar *restrict inp, uint img_w, uint img_h,
                     __global uchar *restrict out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    out[img_w * gid.y + gid.x] = inp[img_w * gid.y + gid.x];
}

__kernel void encode_uv(__global uchar *restrict inp, uint img_w, uint img_h,
                     __global uchar *restrict out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    uint off = img_w * img_h;
//    out[off + img_w * gid.y + gid.x] = inp[off + img_w * gid.y + gid.x];
    out[img_w * gid.y + 2 * gid.x] = inp[ off + img_w * gid.y + 2 * gid.x];
    out[img_w * gid.y + 2 * gid.x + 1] = inp[ off + img_w * gid.y + 2 * gid.x + 1];
}

__kernel void decode(__global uchar *restrict inp, uint img_w,
                     __global uchar *restrict out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    out[img_w * gid.y + gid.x] = inp[img_w * gid.y + gid.x];
}

__kernel void decode_y(__global uchar *restrict inp, uint img_w, uint img_h,
                     __global uchar *restrict out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    out[img_w * gid.y + gid.x] = inp[img_w * gid.y + gid.x];
}

__kernel void decode_uv(__global uchar *restrict inp, uint img_w, uint img_h,
                     __global uchar *restrict out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    uint off = img_w * img_h;
    out[off + img_w * gid.y +  2 * gid.x] = inp[img_w * gid.y + 2 * gid.x];
    out[off + img_w * gid.y + 2 * gid.x +1] = inp[img_w * gid.y + 2 * gid.x + 1];
//    out[img_w * gid.y + gid.x] = inp[img_w * gid.y + gid.x];
}
