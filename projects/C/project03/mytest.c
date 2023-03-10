#include <stdio.h>
#include <assert.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 13 (public13.c)
 *
 * Tests calling is_safe() on a simple program.  Call your
 * disassemble() function on the program, to see what its instructions are.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define PROGRAM_SIZE 3

int main(void) {
  
                              
  Wrd program[PROGRAM_SIZE]= {0xc8200014, 0xc8400006, 0x08650000};
                              
  unsigned short location= -1;


  printf("%d\n",is_safe(program, PROGRAM_SIZE, &location));

  assert(is_safe(program, PROGRAM_SIZE, &location) == 1);

  printf("The assertion worked successfully!\n");

  return 0;
}
