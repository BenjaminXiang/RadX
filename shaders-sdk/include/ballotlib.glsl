#ifndef _BALLOTLIB_H
#define _BALLOTLIB_H

#include "../include/mathlib.glsl"

// for constant maners
#ifndef Wave_Size
    #if (defined(AMD_PLATFORM))
        #define Wave_Size 64u
        #define bqualf highp
    #else
        #define Wave_Size 32u
        #define bqualf lowp
    #endif
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
#define IFALL(b) [[flatten]]if(subgroupAll(b))
#define IFANY(b)            if(subgroupAny(b))

const uint UONE = 1u;
//lowp uvec2 bPrefixSum(in bool val) { return uvec2(subgroupAdd(uint(val)), subgroupExclusiveAdd(uint(val))); };
//lowp uvec2 bPrefixSum() { return uvec2(subgroupAdd(UONE), subgroupExclusiveAdd(UONE)); };

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
    if (subgroupElect()) {atomicAdd(mem, T(pfx.x) * T(by), gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsRelaxed);};\
};

#endif
