


// roundly like http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf
// partially of https://vgc.poly.edu/~csilva/papers/cgf.pdf + and wide subgroup adaptation
// practice maximal throughput: 256x16 (of Nx256)


// 2-bit
//#define BITS_PER_PASS 2
//#define RADICES 4u
//#define RADICES_MASK 0x3u

#ifdef ENABLE_TURING_INSTRUCTION_SET
// 8-bit (risen again, but Turing only)
#define ENABLE_SUBGROUP_PARTITION_SORT
#define BITS_PER_PASS 8
#define RADICES 256u
#define RADICES_MASK 0xFFu
#define READ_U8
#define SHF8B 0
#else
// 4-bit
#define BITS_PER_PASS 4
#define RADICES 16u
#define RADICES_MASK 0xFu
#define SIMPLER_SORT
#define SHF8B 1
#endif


// general work groups
#define Wave_Size_RX Wave_Size_RT
#define Wave_Count_RX Wave_Count_RT 


//#if defined(ENABLE_TURING_INSTRUCTION_SET)
    #define VEC_SIZE 4u
    #define VEC_MULT VEC_SIZE
    #define VEC_SHIF 2u
    #define VEC_SEQU WPTRX(Wave_Idx) // yes, yes!
    #define KTYPE utype_v//utype_t[VEC_SIZE]
    #define ATYPE uvec4

/*
    #define VEC_SIZE 2u
    #define VEC_MULT VEC_SIZE
    #define VEC_SHIF 1u
    #define VEC_SEQU WPTRX(Wave_Idx) // yes, yes!
    #define KTYPE utype_v//utype_t[VEC_SIZE]
    #define ATYPE uvec2
*/

    #define WPTRX uint
    #define BOOLX bool

// 
#ifdef ENABLE_SUBGROUP_PARTITION_SORT
#define Wave_Count VEC_SIZE
#else
#define Wave_Count 16u
#endif

// default values
#ifndef BLOCK_SIZE
#define BLOCK_SIZE (Wave_Size*Wave_Count)
#endif


#define BLOCK_SIZE_RT (gl_WorkGroupSize.x)
#define WRK_SIZE_RT (gl_NumWorkGroups.y * Wave_Count_RX)


#define PREFER_UNPACKED
#define utype_t u8x1_t
#define utype_v u8x4_t

#ifdef USE_MORTON_32
#define KEYTYPE uint32_t
//lowp uint BFE(in uint32_t ua, in int o, in int n) { return bitfieldExtract(ua, o, n); }
#define BFE bitfieldExtract
#else
#define KEYTYPE u32vec2
lowp uint BFE(in u32vec2 ua, in int o, in int n) { return uint(o >= 32 ? bitfieldExtract(ua.y, o-32, n) : bitfieldExtract(ua.x, o, n)); }
#endif

struct RadicePropStruct { uint Descending, IsSigned; };

#ifdef COPY_HACK_IDENTIFY
#define INDIR 1
#define OUTDIR 0
#else
#define INDIR 0
#define OUTDIR 1
#endif

// used when filling
const KEYTYPE OutOfRange = KEYTYPE(0xFFFFFFFFu);

layout ( binding = 0, set = INDIR, scalar )  readonly subgroupcoherent buffer KeyInB {KEYTYPE KeyIn[]; };
layout ( binding = 1, set = INDIR, scalar )  readonly subgroupcoherent buffer ValueInB {uint ValueIn[]; };
layout ( binding = 0, set = OUTDIR, scalar )  subgroupcoherent buffer KeyTmpB {KEYTYPE KeyTmp[]; };
layout ( binding = 1, set = OUTDIR, scalar )  subgroupcoherent buffer ValueTmpB {uint ValueTmp[]; };

// 
layout ( binding = 3, set = 0, scalar )  subgroupcoherent buffer HistogramB {uint Histogram[][RADICES]; };
layout ( binding = 4, set = 0, scalar )  subgroupcoherent buffer PrefixSumB {uint PrefixSum[][RADICES]; };

// push constant in radix sort
layout ( push_constant ) uniform PushBlock { uint Shift, r0, r1, r2; } push_block;
layout ( binding = 6, set = 0, scalar ) uniform InlineUniformB { uint data; } internal_block[];
layout ( binding = 6, set = 1, scalar ) uniform InputInlineUniformB { uint data; } inline_block[];

#define NumElements inline_block[0].data

// division of radix sort (TODO: fix corruptions)
struct blocks_info { uint count, limit, offset, wkoffset; };
blocks_info get_blocks_info(in uint n) {
    const uint 
        block_tile = Wave_Size_RT << VEC_SHIF, 
        block_size_per_work = tiled(n, gl_NumWorkGroups.x), 
        block_size = tiled(block_size_per_work, block_tile) * block_tile, 
        block_offset = block_size * gl_WorkGroupID.x,
        block_limit = block_offset + block_size,
        block_count = tiled(block_size, block_tile),
        block_offset_single = (block_size>>VEC_SHIF)*gl_WorkGroupID.x;

    return blocks_info(block_count, min(block_limit, n), block_size*gl_WorkGroupID.x, (block_size>>VEC_SHIF)*gl_WorkGroupID.x);
};

#ifdef PREFER_UNPACKED
#define upfunc(x) (x)
#else
#define upfunc(x) up2x_8(x)
#endif

#ifndef WORK_SIZE
#define WORK_SIZE BLOCK_SIZE
#endif
