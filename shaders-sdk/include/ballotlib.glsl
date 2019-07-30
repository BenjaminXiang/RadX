#ifndef _BALLOTLIB_H
#define _BALLOTLIB_H

#include "../include/mathlib.glsl"

// for constant maners
#if (defined(AMD_PLATFORM))
    #define bqtype_t uint64_t
    #define bqtype2 u64vec2
    #define Wave_Size 64u
    #define bqualf highp
    bqtype_t extblt(in uvec4 blt){return pack64(blt.xy);};
    bqtype2 extbl2(in uvec4 blt){return bqtype2(pack64(blt.xy),pack64(blt.zw));};
    bool bltinv(in bqtype_t a){return subgroupInverseBallot(uvec4(unpack32(a),0u.xx));};
#else
    #define bqtype_t uint16_t
    #define bqtype2 uint32_t
    #define bqtype4 uint64_t
    #define Wave_Size 16u//32u
    #define bqualf lowp
    bqtype_t extblt(in uvec4 blt){ return unpack16(blt.x)[gl_SubgroupInvocationID>>4u]; };
    bqtype2 extbl2(in uvec4 blt){ return blt.x; };
    bool bltinv2(in bqtype2 a){ return subgroupInverseBallot(uvec4(a,0u.xxx)); };
    bool bltinv(in bqtype_t a){
        u16vec2 m16 = u16vec2(0u); m16[gl_SubgroupInvocationID>>4u] = a;
        return subgroupInverseBallot(uvec4(pack32(m16),0u.xxx));
    };
#endif


// Z-swizzle for 
#define Ni 2u

#ifdef IS_RAY_SHADER
#define WID gl_LaunchIDNV[Ni].x // shuld be uniform (scalar)
#else
#define WID gl_GlobalInvocationID[Ni].x // shuld be uniform (scalar)
#endif

#define Wave_Size_RT gl_SubgroupSize
#define Wave_Count_RT gl_NumSubgroups

#ifndef OUR_INVOC_TERM
    #define Launch_Idx gl_GlobalInvocationID
    #define Local_Idx gl_LocalInvocationIndex
    #define Wave_Idx gl_SubgroupID
    #define Lane_Idx gl_SubgroupInvocationID
#endif

// 
#define uint_ballot uvec4
#define readLane subgroupBroadcast
#define readFLane subgroupBroadcastFirst
#define electedInvoc subgroupElect

// subgroup barriers
//#define LGROUP_BARRIER subgroupBarrier();//memoryBarrier(),subgroupBarrier();
#define LGROUP_BARRIER subgroupBarrier();
#define IFALL(b) [[flatten]]if(subgroupAll(b)) // TODO: support 16-bit subgroups 
#define IFANY(b)            if(subgroupAny(b)) // TODO: support 16-bit subgroups 

const uint UONE = 1u;

lowp uvec2 bPrefixSum() {
    const bqualf uvec4 ballot = subgroupBallot(true);
    return uvec2(subgroupBallotBitCount(ballot), subgroupBallotExclusiveBitCount(ballot));
    //return uvec2(subgroupAdd(UONE), subgroupExclusiveAdd(UONE)); 
};

lowp uint bSum() {
    return subgroupBallotBitCount(subgroupBallot(true));
};

#define initAtomicSubgroupIncFunction(mem, fname, by, T)\
T fname() {\
    const lowp uvec2 pfx = bPrefixSum();\
    T gadd = 0; [[flatten]] if (subgroupElect()) {gadd = atomicAdd(mem, T(pfx.x) * T(by));}; gadd = readFLane(gadd);\
    return T(pfx.y) * T(by) + gadd;\
};

#define initAtomicSubgroupIncFunctionTarget(mem, fname, by, T)\
T fname(in  uint WHERE) {\
    const lowp uvec2 pfx = bPrefixSum();\
    T gadd = 0; [[flatten]] if (subgroupElect()) {gadd = atomicAdd(mem, T(pfx.x) * T(by));}; gadd = readFLane(gadd);\
    return T(pfx.y) * T(by) + gadd;\
};

#define initAtomicSubgroupIncFunctionTargetBinarity(mem, fname, by, T)\
T fname(in  uint WHERE) {\
    const lowp uvec2 pfx = bPrefixSum();\
    T gadd = 0; [[flatten]] if (subgroupElect()) {gadd = atomicAdd(mem[WID], T(pfx.x) * T(by));}; gadd = readFLane(gadd);\
    return T(pfx.y) * T(by) + gadd;\
};

// statically multiplied
#define initSubgroupIncFunctionTarget(mem, fname, by, T)\
T fname(in  uint WHERE) {\
    const lowp uvec2 pfx = bPrefixSum();\
    T gadd = 0; [[flatten]] if (subgroupElect()) {gadd = atomicAdd(mem, T(pfx.x) * T(by), gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsRelaxed);}; gadd = readFLane(gadd);\
    return T(pfx.y) * T(by) + gadd;\
};

// statically multiplied
#define initSubgroupIncReducedFunctionTarget(mem, fname, by, T)\
void fname(in  uint WHERE) {\
    const lowp uint pfx = bSum();\
    [[flatten]] if (subgroupElect()) {atomicAdd(mem, T(pfx.x) * T(by), gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsRelaxed);};\
};


// 
uint16_t sgrblt(in bool k) { return unpack16(subgroupBallot(k).x)[gl_SubgroupInvocationID>>4u]; };
uint16_t sgrprt(in lowp uint k) { return unpack16(subgroupPartitionNV(k).x)[gl_SubgroupInvocationID>>4u]; };
uint16_t genLtMask() { return unpack16(gl_SubgroupLtMask.x)[gl_SubgroupInvocationID>>4u]; };
uint32_t genLtMask2() { return pack32(u16vec2(gl_LocalInvocationID.y == 0u ? genLtMask() : 0xFFFFFFFFus, gl_LocalInvocationID.y == 1u ? genLtMask() : 0us)); };
//uint16_t genLtMask() { return unpack16(gl_SubgroupLtMask.x)[gl_SubgroupInvocationID>>4u]; };


uint sgrshf(in uint bk, in uint ps) {
//#ifdef CLUSTERED_SUPPORTED
//    return subgroupClusteredShuffle(bk,ps,16u); // GLSL is f&cked language...
//#else
//#ifdef ENABLE_TURING_INSTRUCTION_SET

//#else
    return subgroupShuffle(bk,(ps&15u)|((gl_SubgroupInvocationID>>4u)<<4u));
//#endif
//    return subgroupShuffle(bk,ps); // SERIOSLY?! WHERE SUBGROUP SIZE CONTROL?
};

uint sgrsumex(in uint bk) {
//#ifdef CLUSTERED_SUPPORTED
//    return subgroupClusteredExclusiveAdd(bk,Wave_Size); // GLSL is f&cked language...
//#else
#ifdef ENABLE_TURING_INSTRUCTION_SET
    return subgroupPartitionedExclusiveAddNV(bk,uvec4(0xFFFFu<<((gl_SubgroupInvocationID>=Wave_Size?1u:0u)*16u),0u.xxx)); // Wallhack 
#else
    uint sm = 0u;
    [[flatten]] if (gl_SubgroupInvocationID < Wave_Size) sm = subgroupExclusiveAdd(gl_SubgroupInvocationID < Wave_Size ? bk : 0u); // blurry part I
    [[flatten]] if (gl_SubgroupInvocationID >=Wave_Size) sm = subgroupExclusiveAdd(gl_SubgroupInvocationID >=Wave_Size ? bk : 0u); // blurry part II
    return sm;
#endif
//    return subgroupExclusiveAdd(bk); // SERIOSLY?! WHERE SUBGROUP SIZE CONTROL?
}

#endif
