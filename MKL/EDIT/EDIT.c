#include <stdio.h>
#include "stdlib.h"
#include "string.h"
#include "math.h"

int min(int x, int y, int z){
    return min(min(x, y), z);
}

struct Sequence{
    int length;
    char *seq;
};
 
int EDIT(char *str1, int m, char *str2, int n){
    int dp[m+1][n+1];

    for (int i=0; i<=m; i++){
        for (int j=0; j<=n; j++){
            if (i==0)
                dp[i][j] = j;
            else if (j==0)
                dp[i][j] = i;
            else if (str1[i-1] == str2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]);
        }
    }
    return dp[m][n];
}


void main(int argc, char *argv[]){
    int i, j;
    FILE *fp;
    float v, edit_dist[70000][70000];
    struct Sequence s[70000];
    char *line = malloc(1000);
    fp = fopen(argv[1], "r");
    for(i=0; i < 70000; i++)
        edit_dist[i][i] = 0.0;

    j = 0;
    while(fgets(line, 1000, fp) != NULL){
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

    #pragma omp parallel for
    for(i=0; i<70000; i++){
        for(j=i+1; j<70000; j++){
            edit_dist[i][j] = EDIT(s[i].distance, s[i].length, s[j].distance, s[j].length);
            edit_dist[j][i] = edit_dist[i][j];
        }
    }
}
