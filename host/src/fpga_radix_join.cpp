/** \copydoc RJ */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>              /* CPU_ZERO, CPU_SET */
#include <pthread.h>            /* pthread_* */
#include <stdlib.h>             /* malloc, posix_memalign */
#include <sys/time.h> 
#include <unistd.h>             /* gettimeofday */
#include <stdio.h>              /* printf */
#include <iostream>
//#include <smmintrin.h>          /* simd only for 32-bit keys – SSE4.1 */

#include "fpga_radix_join.h"
#include "cpu_parallel_radix_join.h"
#include "prj_params.h"  

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)

#ifndef NEXT_POW_2
#define NEXT_POW_2(V)                           \
	do {                                        \
		V--;                                    \
		V |= V >> 1;                            \
		V |= V >> 2;                            \
		V |= V >> 4;                            \
		V |= V >> 8;                            \
		V |= V >> 16;                           \
		V++;                                    \
	} while(0)
#endif

void dump_error(const char *str, cl_int status) {
	printf("%s\n", str);
	printf("Error code: %d\n", status);
}

void cleanup(){
}

int creatKernels(){
	kernel_relRead = clCreateKernel(program, "relRead", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	} 
	kernel_gather = clCreateKernel(program, "gather", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}

	kernel_filter[0] = clCreateKernel(program, "filter", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_filter[1] = clCreateKernel(program, "filter1", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_filter[2] = clCreateKernel(program, "filter2", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_filter[3] = clCreateKernel(program, "filter3", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_filter[4] = clCreateKernel(program, "filter4", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_filter[5] = clCreateKernel(program, "filter5", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_filter[6] = clCreateKernel(program, "filter6", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_filter[7] = clCreateKernel(program, "filter7", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}

#if NUM_FPGA_DATAPATH > 8
	kernel_filter[8] = clCreateKernel(program, "filter8", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 9
	kernel_filter[9] = clCreateKernel(program, "filter9", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 10
	kernel_filter[10] = clCreateKernel(program, "filter10", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 11
	kernel_filter[11] = clCreateKernel(program, "filter11", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 12
	kernel_filter[12] = clCreateKernel(program, "filter12", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 13
	kernel_filter[13] = clCreateKernel(program, "filter13", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 14
	kernel_filter[14] = clCreateKernel(program, "filter14", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 15
	kernel_filter[15] = clCreateKernel(program, "filter15", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
/*
	kernel_sgather = clCreateKernel(program, "sgather", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[0] = clCreateKernel(program, "sfilter", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[1] = clCreateKernel(program, "sfilter1", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[2] = clCreateKernel(program, "sfilter2", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[3] = clCreateKernel(program, "sfilter3", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[4] = clCreateKernel(program, "sfilter4", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[5] = clCreateKernel(program, "sfilter5", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[6] = clCreateKernel(program, "sfilter6", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
	kernel_sfilter[7] = clCreateKernel(program, "sfilter7", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}

#if NUM_FPGA_DATAPATH > 8
	kernel_sfilter[8] = clCreateKernel(program, "sfilter8", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 9
	kernel_sfilter[9] = clCreateKernel(program, "sfilter9", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 10
	kernel_sfilter[10] = clCreateKernel(program, "sfilter10", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 11
	kernel_sfilter[11] = clCreateKernel(program, "sfilter11", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 12
	kernel_sfilter[12] = clCreateKernel(program, "sfilter12", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 13
	kernel_sfilter[13] = clCreateKernel(program, "sfilter13", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 14
	kernel_sfilter[14] = clCreateKernel(program, "sfilter14", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
#if NUM_FPGA_DATAPATH > 15
	kernel_sfilter[15] = clCreateKernel(program, "sfilter15", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	}
#endif
*/
	kernel_hashjoin = clCreateKernel(program, "hashjoin", &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateKernel.", status);
		freeResources();
		return 1;
	} 
	//varMap();
	return 0;
}

/*
int varMap(){
	bool status;
	status = clEnqueueSVMMap(queue_relRead, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)rTable, rTableSize, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMMap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMMap(queue_relRead, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
			(void *)sTable, sTableSize, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMMap", status);
		freeResources();
		return 1;
	}   
	status = clEnqueueSVMMap(queue_relRead, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
			(void *)rTableReadRange, sizeof(unsigned int) * 2, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMMap", status);
		freeResources();
		return 1;
	}  
	status = clEnqueueSVMMap(queue_relRead, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
			(void *)sTableReadRange, sizeof(unsigned int) * 2, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMMap", status);
		freeResources();
		return 1;
	}  
	status = clEnqueueSVMMap(queue_hashjoin, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
			(void *)matchedTable, matchedTableSize, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMMap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMMap(queue_hashjoin, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
			(void *)sTableReadRange, sizeof(unsigned int) * 2, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMMap", status);
		freeResources();
		return 1;
	}  
	status = clEnqueueSVMMap(queue_hashjoin, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
			(void *)rTableReadRange, sizeof(unsigned int) * 2, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMMap", status);
		freeResources();
		return 1;
	} 
	return 0;
}
int varUnmap(){
	bool status;
	status = clEnqueueSVMUnmap(queue_relRead, (void *)rTable, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMUnmap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMUnmap(queue_relRead, (void *)rTableReadRange, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMUnmap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMUnmap(queue_relRead, (void *)sTableReadRange, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMUnmap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMUnmap(queue_relRead, (void *)sTable, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMUnmap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMUnmap(queue_hashjoin, (void *)matchedTable, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMUnmap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMUnmap(queue_hashjoin, (void *)sTableReadRange, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMUnmap", status);
		freeResources();
		return 1;
	}
	status = clEnqueueSVMUnmap(queue_hashjoin, (void *)rTableReadRange, 0, NULL, NULL); 
	if(status != CL_SUCCESS) {
		dump_error("Failed clEnqueueSVMUnmap", status);
		freeResources();
		return 1;
	}
	return 0;
}
*/

void freeResources() {
	if(kernel_hashjoin) 
		clReleaseKernel(kernel_hashjoin);
	if(kernel_relRead)
		clReleaseKernel(kernel_relRead);    
	if(program) 
		clReleaseProgram(program);
	if(queue_relRead) 
		clReleaseCommandQueue(queue_relRead);
	if(queue_hashjoin) 
		clReleaseCommandQueue(queue_hashjoin);
	if(rTable) 
		clSVMFreeAltera(context,rTable);
	if(sTable) 
		clSVMFreeAltera(context,sTable);
	if(matchedTable) 
		clSVMFreeAltera(context,matchedTable);
	if(context) 
		clReleaseContext(context);
}

int setHardwareEnv(
		cl_uint &num_platforms,
		cl_uint &num_devices
		){
	status = clGetPlatformIDs(1, &platform, &num_platforms);
	if(status != CL_SUCCESS) {
		dump_error("Failed clGetPlatformIDs.", status);
		freeResources();
		return 1;
	}
	if(num_platforms != 1) {
		printf("Found %d platforms!\n", num_platforms);
		freeResources();
		return 1;
	}
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
	if(status != CL_SUCCESS) {
		dump_error("Failed clGetDeviceIDs.", status);
		freeResources();
		return 1;
	}
	if(num_devices != 1) {
		printf("Found %d devices!\n", num_devices);
		freeResources();
		return 1;
	}
	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateContext.", status);
		freeResources();
		return 1;
	}
	return 0;
}

int64_t fpga_hash_join(const relation_t * tmpR, const relation_t * tmpS, uint offset_r, uint offset_s){

	//uint * rTable = (uint *)tmpR->tuples;
	uint rTupleNum = tmpR-> num_tuples;

	//uint * sTable = (uint *)tmpS->tuples;
	uint sTupleNum = tmpS-> num_tuples;

	rTableReadRange[0] = offset_r;

	rTableReadRange[1] = rTupleNum;

	sTableReadRange[0] = offset_s;

	sTableReadRange[1] = sTupleNum;
	// assign the read offset here 
	//varMap();
	printf("rtuple offset %d  num is %d , stuple offset %d num is %d \n",offset_r, rTupleNum, offset_s, sTupleNum);
	//printf("Launching the hash join kernel...\n");

	cl_event event_hashjoin, event_relRead;
	//printf("checkpoint \n");
	status = clEnqueueTask(queue_relRead, kernel_relRead, 0, NULL, &event_relRead);
	if (status != CL_SUCCESS) {
		dump_error("Failed to launch kernel.", status);
		return 1;
	}
	status = clEnqueueTask(queue_hashjoin, kernel_hashjoin, 0, NULL, &event_hashjoin);
	if (status != CL_SUCCESS) {
		dump_error("Failed to launch kernel.", status);
		return 1;
	}

	clFinish(queue_relRead);
	clFinish(queue_hashjoin);

	//printf("result %d \n", matchedTable[0]);
	return matchedTable[0];
}


int64_t
RJ_FPGA(relation_t * relR, relation_t * relS, int nthreads)
{
	int64_t result = 0;
	uint32_t i;

	relation_t *outRelR, *outRelS;

	outRelR = (relation_t*) malloc(sizeof(relation_t));
	outRelS = (relation_t*) malloc(sizeof(relation_t));

	/* allocate temporary space for partitioning */
	/* TODO: padding problem */
	size_t sz = relR->num_tuples * sizeof(tuple_t) + RELATION_PADDING;
	outRelR->tuples     = (tuple_t*) malloc(sz);
	outRelR->num_tuples = relR->num_tuples;

	sz = relS->num_tuples * sizeof(tuple_t) + RELATION_PADDING;
	outRelS->tuples     = (tuple_t*) malloc(sz);
	outRelS->num_tuples = relS->num_tuples;

	/***** do the multi-pass partitioning *****/
#if NUM_PASSES==1
	/* apply radix-clustering on relation R for pass-1 */
	radix_cluster_nopadding(outRelR, relR, 0, NUM_RADIX_BITS_FPGA);
	relR = outRelR;

	/* apply radix-clustering on relation S for pass-1 */
	radix_cluster_nopadding(outRelS, relS, 0, NUM_RADIX_BITS_FPGA);
	relS = outRelS;

#elif NUM_PASSES==2
	/* apply radix-clustering on relation R for pass-1 */
	radix_cluster_nopadding(outRelR, relR, 0, NUM_RADIX_BITS_FPGA/NUM_PASSES);

	/* apply radix-clustering on relation S for pass-1 */
	radix_cluster_nopadding(outRelS, relS, 0, NUM_RADIX_BITS_FPGA/NUM_PASSES);

	/* apply radix-clustering on relation R for pass-2 */
	radix_cluster_nopadding(relR, outRelR,
			NUM_RADIX_BITS_FPGA/NUM_PASSES, 
			NUM_RADIX_BITS_FPGA-(NUM_RADIX_BITS_FPGA/NUM_PASSES));

	/* apply radix-clustering on relation S for pass-2 */
	radix_cluster_nopadding(relS, outRelS,
			NUM_RADIX_BITS_FPGA/NUM_PASSES, 
			NUM_RADIX_BITS_FPGA-(NUM_RADIX_BITS_FPGA/NUM_PASSES));

	/* clean up temporary relations */
	free(outRelR->tuples);
	free(outRelS->tuples);
	free(outRelR);
	free(outRelS);

#else
#error Only 1 or 2 pass partitioning is implemented, change NUM_PASSES!
#endif

	int * R_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS_FPGA), sizeof(int));
	int * S_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS_FPGA), sizeof(int));

	/* compute number of tuples per cluster */
	for( i=0; i < relR->num_tuples; i++ ){
		uint32_t idx = (relR->tuples[i].key) & ((1<<NUM_RADIX_BITS_FPGA)-1);
		R_count_per_cluster[idx] ++;
	}
	for( i=0; i < relS->num_tuples; i++ ){
		uint32_t idx = (relS->tuples[i].key) & ((1<<NUM_RADIX_BITS_FPGA)-1);
		S_count_per_cluster[idx] ++;
	}

	/* build hashtable on inner */
	int r, s; /* start index of next clusters */
	r = s = 0;
	int offset_r, offset_s;
	offset_r = offset_s = 0;
	for( i=0; i < (1<<NUM_RADIX_BITS_FPGA); i++ ){
		relation_t tmpR, tmpS;

		if(R_count_per_cluster[i] > 0 && S_count_per_cluster[i] > 0){

			tmpR.num_tuples = R_count_per_cluster[i];
			tmpR.tuples = relR->tuples + r;
			offset_r = r;
			r += R_count_per_cluster[i];

			tmpS.num_tuples = S_count_per_cluster[i];
			tmpS.tuples = relS->tuples + s;
			offset_s = s;
			s += S_count_per_cluster[i];

			result += fpga_hash_join(&tmpR, &tmpS, offset_r, offset_r);
		}
		else {
			r += R_count_per_cluster[i];
			s += S_count_per_cluster[i];
		}
	}

	/* clean-up temporary buffers */
	free(S_count_per_cluster);
	free(R_count_per_cluster);

#if NUM_PASSES == 1
	/* clean up temporary relations */
	free(outRelR->tuples);
	free(outRelS->tuples);
	free(outRelR);
	free(outRelS);
#endif

	return result;
}

