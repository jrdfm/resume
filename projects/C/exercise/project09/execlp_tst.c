#include "safe-fork.h"
#include <stdio.h>
#include <stdlib.h>
#include <wait.h>
#include <sys/types.h>
#include <unistd.h>
#include <err.h>
#include <sysexits.h>

/* (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
   not to provide it to anyone else. */

/* Illustrates a program that forks a child, and has the child exec another
 * program (the ls program in this case).  The parent waits for the child to
 * terminate, then does some other processing afterward.  (OK, the other
 * processing is not all that exciting, but it demonstrates the point....)
 */

int main(void) {
  pid_t child_pid;
  char *arr[] = {"gcc", "execlp_tst.c", "safe-fork.o" ,"-o" ,"tst_.x",NULL}; 

  child_pid= safe_fork();

  if (child_pid > 0) {

    int status;
    wait(&status);  /* we  care about the child's exit status */
    printf("\nI am the parent.  My child was %d.\n", child_pid);
    
    if (WIFEXITED(status))
      printf("Child exited with status %d.\n", WEXITSTATUS(status));
    else printf("Child was seriously borked; did not terminate normally.\n");

  } else
    if (child_pid == 0) {  /* this is the child code */

      printf("I am the child (%d).\n", getpid());
      printf("Now replacing myself with \"gcc execlp_tst.c safe-fork.o -o tst_.x\".\n\n");

       execvp(arr[0],arr);
      /*execlp("gcc", "gcc", "execlp_tst.c","safe-fork.o", "-o", "_tst.x", NULL);
       */
     exit(2);

    } else err(EX_OSERR, "fork error");  /* error case; child_pid == -1 */

  return 0;
}
