#include "safe-fork.h"
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sysexits.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  char *pgm;
  pid_t pid, pid2;
  int pipefd[2];
  int wtcomm, wcomm;
  pgm = "./remove-comments.x";
  if (argc != 2) {
    exit(2);
  }
  pipe(pipefd);
  pid = safe_fork();

  if (pid > 0) { /* parent */

    dup2(pipefd[0], STDIN_FILENO);

    scanf("%d", &wtcomm);
    pid2 = safe_fork();
    if (pid2 == 0) { /*second child */

      dup2(pipefd[1], STDOUT_FILENO);
      execl("/bin/wc", "wc", "-m", argv[1], NULL);

    } else {
      scanf("%d", &wcomm);
      if (wcomm >= (2 * wtcomm)) {
        printf("Enough comments!\n");
        exit(0);
      } else
        printf("Too few comments!\n");
      exit(1);
    }
  } else

      if (pid == 0) { /* child */
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);
    execvp(pgm, argv);

    err(EX_OSERR, "execvp error");
  } else
    err(EX_OSERR, "fork error");
  return 0;
}
