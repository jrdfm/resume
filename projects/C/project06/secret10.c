#include <stdio.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 12 (secret12.c)
 *
 * Tests passing NULL into init().
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
 /* this should have no effect, and definitely not crash */
  init(NULL);

  printf("If this is the only thing printed by this test- and of course if ");
  printf("it didn't\ncrash- then it worked successfully!\n");

  return 0;
}
