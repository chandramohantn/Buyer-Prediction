#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "math.h"

struct Sequence{
	int length;
	float *distance;
};

float DTW(float *v1, int N,  float *v2, int M){
	int i, j, k, I, **track;
	float **distance, **D, returnValue, min; 
	float top, bottom, mid;

	distance = (float **) calloc (N, sizeof(float*));
	for (i = 0; i < N; i++)
		distance[i] = (float *) calloc (M, sizeof(float));
	 
	for (i = 0; i < N; i++){
		for (j = 0; j < M; j++){
			distance[i][j] = abs(*(v1 + i) - *(v2 + j));
		}
	}
	 
	D = (float **) calloc (N, sizeof(float*));
	track = (int **) calloc (N, sizeof(int *));
 
	for (i = 0; i < N; i++){
		D[i] = (float *) calloc (M, sizeof(float));
		track[i] = (int *) calloc (M, sizeof(int));
	}

	for (i = 0; i < N; i++){
		for (j = 0; j < M; j++){
			track[i][j]=0;
		}
	}

	 D[0][0] = distance[0][0];
	 for (i = 1; i < N; i++)
	 	D[i][0] = distance[i][0] + D[i-1][0];
	 for (j = 1; j < M; j++)
		 D[0][j] = distance[0][j] + D[0][j-1];
	 
	 for (i = 1; i < N; i++){
		for (j = 1; j < M; j++){
			top = D[i][j-1];
			mid = D[i-1][j-1];
			bottom = D[i-1][j];

			if(top < mid && top < bottom){
				min = top;
				I = 3;
			}
			else if(mid < bottom){
				min = mid;
				I=2;
			}
			else{
				min = bottom;
				I=1;
			}
			D[i][j] = distance[i][j] + min;
			track[i][j] = I;
		}
	}
	
	returnValue = D[N - 1][M - 1];
	free(distance);
	free(D);
	return returnValue;
	}

void main(){
    int i, j;
    FILE *fp = fopen("./Train2.txt", "r");
    float *v, dtw_dist[70000][70000];
    struct Sequence s[70000];
    char line[5000];
    //line = (char *)malloc(2000);
    printf("Open file pointer");
	//for(i=0; i < 70000; i++)
	//	dtw_dist[i][i] = 0.0;

	j = 0;
	char *item;
	printf("start read ....");
	while(fgets(line, sizeof(line), fp)){
		printf("%d\n", sizeof(line));
		printf("%d\n", j+1);
		item = strtok(line, ",");
		int c = atoi(item);
		v = (float *)malloc(c * sizeof(float));
		i = 0;
		while(item != NULL){
			item = strtok(NULL, ",");
			v[i] = atof(item);
			i++;
		}
		s[j].length = c;
		s[j].distance = v;
		j++;
	}
	fclose(fp);
	printf("File read complete .....");

	//#pragma omp parallel for
	for(i=0; i<70000; i++){
		printf("%d\n", i+1);
		for(j=i+1; j<70000; j++){
			dtw_dist[i][j] = 1.0 / DTW(s[i].distance, s[i].length, s[j].distance, s[j].length);
			dtw_dist[j][i] = dtw_dist[i][j];
		}
	}

	fp = fopen("DTW_kernel.csv", "w");
	for(i=0; i<70000; i++)
		for(j=0; j<70000; j++)
			fprintf(fp, "%f,", dtw_dist[i][j]);
	fclose(fp);
}

//export OMP_NUM_THREADS=2
