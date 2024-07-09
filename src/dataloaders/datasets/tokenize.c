#include <stdlib.h>
#include <stdio.h>

/*
  basemap[] maps a nucleotide character to a specific integer value:
  A -> 7
  C -> 8
  G -> 9
  T -> 10
  N -> 11 (default for unknown characters)
*/

static const int basemap[256] = {
    [0 ... 255] = 11,  // Default value for unknown characters
    ['A'] = 0,
    ['C'] = 1,
    ['G'] = 2,
    ['T'] = 3,
    ['N'] = 4
};

int main(int argc, const char** argv)
{
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    
    while ((read = getline(&line, &len, stdin)) != -1) {
        for (int idx = 0; idx < read - 1; ++idx) {
            // Output the mapped integer value without spaces
            printf("%d", basemap[(unsigned char)line[idx]]);
        }
        printf("\n");
    }
    if (line) {
        free(line); 
        line = NULL;
    }
    
    return EXIT_SUCCESS;
}
