#include "mapperInt.h"
#include <stdlib.h>

ABC_NAMESPACE_IMPL_START

typedef struct Map_SupergateUsage_t_ {
    char* truthTableHex;
    int usageCount;
    struct Map_SupergateUsage_t_* next;
} Map_SupergateUsage_t;

static Map_SupergateUsage_t* s_usageList = NULL;
static char* s_jsonPath = NULL;

static char* Map_UsageLogTruthToHex(unsigned* pTruth, int nVars) {
    int nWords = (nVars <= 5) ? 1 : (1 << (nVars - 5));
    char* hex = ABC_ALLOC(char, nWords * 8 + 3);
    char* ptr = hex;
    sprintf(ptr, "0x");
    ptr += 2;
    for (int i = 0; i < nWords; i++) {
        sprintf(ptr, "%08x", pTruth[i]);
        ptr += 8;
    }
    return hex;
}

void Map_UsageLogInit() {
    char* envPath = getenv("SUPERGATE_USAGE_LOG");
    if (!envPath) return;
    s_jsonPath = ABC_ALLOC(char, strlen(envPath) + 1);
    strcpy(s_jsonPath, envPath);
    s_usageList = NULL;
}

void Map_UsageLogUsage(Map_Super_t* pSuper) {
    if (!pSuper || !s_jsonPath) return;
    char* truthHex = Map_UsageLogTruthToHex(pSuper->uTruth, pSuper->nFanins);
    
    Map_SupergateUsage_t* curr = s_usageList;
    while (curr) {
        if (strcmp(curr->truthTableHex, truthHex) == 0) {
            curr->usageCount++;
            ABC_FREE(truthHex);
            return;
        }
        curr = curr->next;
    }
    
    Map_SupergateUsage_t* newEntry = ABC_ALLOC(Map_SupergateUsage_t, 1);
    newEntry->truthTableHex = truthHex;
    newEntry->usageCount = 1;
    newEntry->next = s_usageList;
    s_usageList = newEntry;
}

void Map_UsageLogWriteAndCleanup() {
    if (!s_usageList || !s_jsonPath) return;
    
    FILE* logFile = fopen(s_jsonPath, "w");
    if (logFile) {
        Map_SupergateUsage_t* curr = s_usageList;
        while (curr) {
            fprintf(logFile, "{\"supergate_id\": \"%s\", \"count\": %d}\n", 
                    curr->truthTableHex, curr->usageCount);
            curr = curr->next;
        }
        fclose(logFile);
    }
    
    Map_SupergateUsage_t* curr = s_usageList;
    while (curr) {
        Map_SupergateUsage_t* next = curr->next;
        ABC_FREE(curr->truthTableHex);
        ABC_FREE(curr);
        curr = next;
    }
    s_usageList = NULL;
    ABC_FREE(s_jsonPath);
    s_jsonPath = NULL;
}

ABC_NAMESPACE_IMPL_END