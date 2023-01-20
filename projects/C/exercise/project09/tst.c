#include <stdio.h>
#include "simple-sserver.h"
#include "split.h"

int main(void) {
  Commands commands;
  Node *curr;
  char **splt;
  int i;
  commands= read_commands("public03.compile-cmds", "public03.test-cmds");

  curr = commands.cmp_head;
    
    

 while (curr != NULL)
 {
   splt = split(curr->data);
   i = 0;
 
  printf("%s\n",curr->data);

    while (splt[i] != NULL)
    {
      printf("%s%s",splt[i],",");
      i++;
      
    }
    
    curr = curr->next;
    printf("%s","\n\n");
 }

printf("%s","\n");


 curr = commands.tst_head;
  while (curr != NULL)
 {
   splt = split(curr->data);
   i = 0;
 
  printf("%s\n",curr->data);

    while (splt[i] != NULL)
    {
      printf("%s%s",splt[i],",");
      i++;
      
    }
    
    curr = curr->next;
    printf("%s","\n\n");
 }

printf("%s","\n");

printf("%s","\n\n");


    
 i = compile_program(commands);
 printf("%d\n",i);
 i = test_program(commands);
 printf("%d\n",i);
  
  return 0;
}
