/* Kernels for encoding/decoding the segmentation data */

/*
 * encode/decode_bits_N compresses each value of the segmentation mask to N
 * bits, i.e., max. (2^N-1) classes + the last one reserved for no class.
 *
 * TODO: Mapping function is required to translate the classes to the restricted
 * number of classes.
 */

/** Encode 2 input samples into one (max. 15 classes, CR = 2)
    @param inp: segmentation buffer, can be any size as long as the workgroups
   are set accordingly
    @param detections: the output of the dnn with labels and coords, allows up
   to 15 classes.
    @param out: the encoded segmentation buffer.
*/
__kernel void encode_bits_4(__global const uchar *restrict inp,
                            __global const int *restrict detections,
                            __global uchar *restrict out) {

  // create a local scratchpad
  local uchar scratch_pad[257];
  int tid = get_local_size(0) * get_local_id(1) + get_local_id(0);
  const int local_size = get_local_size(0) * get_local_size(1);
  int j = 257;
  int scratch_pad_offset = 0;

  // set values to zero
  do {
    if (tid < j) {
      scratch_pad[scratch_pad_offset + tid] = 0;
    }
    j -= local_size;
    scratch_pad_offset += local_size;
  } while (j > 0);

  barrier(CLK_LOCAL_MEM_FENCE);

  // create the mapping
  j = detections[0];
  scratch_pad_offset = 0;
  do {
    if (tid < j) {
      int i = (scratch_pad_offset + tid);
      int detection = detections[1 + i * 6] + 1;
      scratch_pad[detection] = i + 1;
    }
    j -= local_size;
    scratch_pad_offset += local_size;
  } while (j > 0);

  // sync the local array
  barrier(CLK_LOCAL_MEM_FENCE);

  int img_w = get_global_size(0);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  // img_w = inp_w/2
  int inp_id = img_w * gid.y + gid.x;

  uchar inp_0 = inp[2 * inp_id + 0] + 1;
  uchar inp_1 = inp[2 * inp_id + 1] + 1;

  uchar x0 = scratch_pad[inp_0];
  uchar x1 = scratch_pad[inp_1];

  uchar y = (x1 & 0xf) << 4 | (x0 & 0xf);

  out[inp_id] = y;
}

/**
 * Decode 1 input sample into two (max. 15 classes, CR = 2)
 * @param inp compressed buffer from encode_bits_4
 * @param detections array of detections from the dnn
 * @param no_class the number to put when a pixel does not have a class
 * @param out the decompressed buffer
 */ 
__kernel void decode_bits_4(__global const uchar *restrict inp,
                            __global const int *restrict detections,
                            const int no_class, __global uchar *restrict out) {

  int tid = get_local_size(0) * get_local_id(1) + get_local_id(0);
  const int local_size = get_local_size(0) * get_local_size(1);
  local uchar scratch_pad[16];
  // create the mapping
  int j = detections[0];
  int scratch_pad_offset = 0;
  scratch_pad_offset = 0;
  do {
    if (tid < j) {
      int i = (scratch_pad_offset + tid);
      int detection = detections[1 + i * 6];
      scratch_pad[i + 1] = detection;
    }
    j -= local_size;
    scratch_pad_offset += local_size;
  } while (j > 0);
  if (tid == 0) {
    scratch_pad[0] = no_class;
  }
  // sync the local array
  barrier(CLK_LOCAL_MEM_FENCE);

  int img_w = get_global_size(0);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int idx = img_w * gid.y + gid.x;

  uchar x = inp[idx];

  uchar y0 = x & 0xf;
  uchar y1 = (x >> 4) & 0xf;

  int out_id = img_w * gid.y + gid.x;
  out[2 * out_id + 0] = scratch_pad[y0];
  out[2 * out_id + 1] = scratch_pad[y1];
}

// TODO: Demapping function
// Decode 1 input sample into two (max. 15 classes, CR = 2)
__kernel void decode_bits_4_simplified(__global const uchar *restrict inp,
                                       __global const int *restrict detections,
                                       const int no_class,
                                       __global int *restrict scratch_pad,
                                       __global uchar *restrict out) {
  // int global_id = get_global_id(0) + get_global_id(1);
  int global_id = get_local_id(0) + get_local_id(1);

  // only run this section once for all work items
  if (global_id == 0) {
    int detect_count = detections[0];

    if (detect_count > 15) {
      detect_count = 0;
    }
    // no detection is mapped to 0, so the detections are
    // 1 indexed
    scratch_pad[0] = no_class;
    for (int i = 0; i < detect_count; i++) {
      int detection = detections[1 + i * 6];

      // make sure we are not out of range
      if (0 <= detection && detection < 256) {
        scratch_pad[i + 1] = detection;
      }
    }
  }
  // sync the local array
  barrier(CLK_GLOBAL_MEM_FENCE);

  int img_w = get_global_size(0);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int idx = img_w * gid.y + gid.x;

  uchar x = inp[idx];

  uchar y0 = x & 0xf;
  uchar y1 = (x >> 4) & 0xf;

  int out_id = img_w * gid.y + gid.x;
  out[2 * out_id + 0] = scratch_pad[y0];
  out[2 * out_id + 1] = scratch_pad[y1];
}

/** Encode 2 input samples into one (max. 15 classes, CR = 2)
    @param inp: segmentation buffer, can be any size as long as the workgroups
   are set accordingly
    @param detections: the output of the dnn with labels and coords, allows up
   to 15 classes.
    @param scratchpad: a buffer for internal use to map labels to encoded values
    @param out: the encoded segmentation buffer.
*/
__kernel void encode_bits_4_simplified(__global const uchar *restrict inp,
                                       __global const int *restrict detections,
                                       __global uchar *restrict out) {

  int global_id = get_local_id(0) + get_local_id(1);
  local uchar scratch_pad[257];
  // only run this section once for all work items in workgroup
  if (global_id == 0) {

    // clear scratch_pad
    for (int i = 0; i < 257; i++) {
      scratch_pad[i] = 0;
    }

    int detect_count = detections[0];

    if (detect_count > 15) {
      detect_count = 0;
    }
    // no detection is mapped to 0, so the detections are
    // 1 indexed
    scratch_pad[0] = 0;
    for (int i = 0; i < detect_count; i++) {
      int detection = detections[1 + i * 6] + 1;

      // make sure we are not out of range
      if (0 <= detection && detection < 257) {
        scratch_pad[detection] = i + 1;
      }
    }
  }

  // sync the local array
  barrier(CLK_LOCAL_MEM_FENCE);

  int img_w = get_global_size(0);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  // img_w = inp_w/2
  int inp_id = img_w * gid.y + gid.x;

  uchar inp_0 = inp[2 * inp_id + 0] + 1;
  uchar inp_1 = inp[2 * inp_id + 1] + 1;

  uchar x0 = scratch_pad[inp_0];
  uchar x1 = scratch_pad[inp_1];

  uchar y = (x1 & 0xf) << 4 | (x0 & 0xf);

  out[inp_id] = y;
}

// TODO:
// Encode 8 input samples into three (max. 7 classes, CR = 2.6667)
// Decode 3 input samples into eight (max. 7 classes, CR = 2.6667)

/*
 * Run-length coding (RLE) -- not parallelized
 */

// __kernel void encode_rle(__global uchar *restrict inp, uint inp_size,
//                          __global uchar *restrict out,
//                          __global ulong *restrict out_size) {
//   uint i = 0;
//   ulong iout = 0;

//   while (i < inp_size) {
//     uint count = 1;

//     while ((i + 1 < inp_size) && (inp[i] == inp[i + 1])) {
//       ++i;
//       ++count;
//     }

//     out[iout++] = count;
//     out[iout++] = inp[i];
//     i += 1;
//   }

//   out_size = iout;
// }

// __kernel void
// decode_rle(__global uchar *restrict inp,
//            __global ulong *restrict inp_size __global uchar *restrict out, )
//            {
//   uint i = 0;
//   uint iout = 0;

//   while (i < inp_size) {
//     uint count = inp[i++];
//     uchar val = inp[i++];

//     for (uint j = 0; j < count, ++j) {
//       out[iout++] = val
//     }
//   }
// }
