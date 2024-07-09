#include <stdlib.h>
#include <stdio.h>

/*
  basemap[] works by storing a very small array that maps a base to 
  its complement, by dereferencing the array with the ASCII char's 
  decimal value as the index
  (int) 'A' = 65;
  (int) 'C' = 67;
  (int) 'G' = 71;
  (int) 'T' = 84;
  (int) 'a' = 97;
  (int) 'c' = 99;
  (int) 'g' = 103;
  (int) 't' = 116;
  (int) 'N' = 78;
  (int) 'U' = 85;
  (int) 'u' = 117;
  etc.
  for example: basemap['A'] => basemap[65] => 'T' etc.
*/

static const unsigned char basemap[256] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
     64, 'T', 'V', 'G', 'H', 'E', 'F', 'C', 'D', 'I', 'J', 'M', 'L', 'K', 'N', 'O',
    'P', 'Q', 'Y', 'S', 'A', 'A', 'B', 'W', 'X', 'R', 'Z',  91,  92,  93,  94,  95,
     96, 't', 'v', 'g', 'h', 'e', 'f', 'c', 'd', 'i', 'j', 'm', 'l', 'k', 'n', 'o',
    'p', 'q', 'y', 's', 'a', 'a', 'b', 'w', 'x', 'r', 'z', 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};

int
main(int argc, const char** argv)
{
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    
    while ((read = getline(&line, &len, stdin)) != -1) {
        for (int idx = read - 2; idx >= 0; --idx) {
            fprintf(stdout, "%c", basemap[(int)line[idx]]);
        }
        fprintf(stdout, "\n");
    }
    if (line) {
        free(line); 
        line = NULL;
    }
    
    return EXIT_SUCCESS;
}