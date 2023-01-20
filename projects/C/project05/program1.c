#include <stdio.h>
#include "row-compare.h"

/* (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  int arr1[][5]= {{3, 1, 4, 1, 5}, {2, 6, 5, 3, 5}, {0, 1, 2, 3, 4}};
  int arr2[][5]= {{0, 0, 0, 0, 0}, {2, 6, 5, 3, 5}, {2, 3, 4, 5, 6}};
  int arr3[][5]= {{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {0, 1, 2, 3, 4}};

  printf("%d %d %d\n", row_compare(arr1, arr2, 0), row_compare(arr1, arr2, 1),
                       row_compare(arr1, arr2, 2));
  printf("%d %d %d\n", row_compare(arr1, arr3, 0), row_compare(arr1, arr3, 1),
                       row_compare(arr1, arr3, 2));

  return 0;
}
