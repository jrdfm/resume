/*
 * CMSC216
 * Spring 2021
 * Line Counter
 * Andrew Chen
 *
 * Counts how many lines in your program (command line parameter) total, are
 * only comments, only whitespace, preprocessor directives, and actual code.
 * Lines that are preprocessor or code with comments are not counted as comment
 * lines.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFFER_SIZE 200

static void count_lines_in(const char* filename) {
    char line_buffer[BUFFER_SIZE];
    int total = 0, comments = 0, whitespace = 0, preprocessor = 0;
    int in_comment = 0, next_line_preprocessor = 0;
    FILE* in = fopen(filename, "r");
    if (in == NULL) {
        fprintf(stderr, "Could not open %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    /* read line-by-line */
    while (fgets(line_buffer, BUFFER_SIZE, in) != NULL) {
        int i;
        int whitespace_line = 1, found_comment_delim = 0, partial_comment = 0;
        char c;
        total++;

        for (i = 0; i < BUFFER_SIZE && line_buffer[i] != '\n' &&
                    (c = line_buffer[i]) != '\0';
             i++) {
            /* find preprocessor directives */
            if (next_line_preprocessor || (c == '#' && whitespace_line)) {
                preprocessor++;
                /* no need to examine rest of line, skip to end and see if \
                 * (causing directive to continue to next line) is present */
                for (; i < BUFFER_SIZE && line_buffer[i] != '\n' &&
                       line_buffer[i] != '\0';
                     i++)
                    ;
                if (line_buffer[i - 1] == '\\') {
                    next_line_preprocessor = 1;
                } else {
                    next_line_preprocessor = 0;
                }
            }

            /* find comments; if the part of the line before the comment
             * delimiter is not whitespace, the line is only partially a comment
             * and contains some code, so in_comment is toggled but the line is
             * not counted towards comment-only lines */
            if (i < BUFFER_SIZE - 1) {
                if (!in_comment && c == '/' && line_buffer[i + 1] == '*') {
                    /* comment start */
                    in_comment = 1;
                    found_comment_delim = 1;
                    /* check for code before comment start */
                    if (!whitespace_line)
                        partial_comment = 1;
                } else if (in_comment && c == '*' && line_buffer[++i] == '/') {
                    /* comment end */
                    in_comment = 0;
                    found_comment_delim = 1;
                    /* skip two chars (once by ++i above and once by continue)
                     * so / is not read as non-whitespace after the comment
                     * delim */
                    /* in case end comment delim is the only content of the
                     * line, it will not be recognized as non-whitespace without
                     * changing that here */
                    whitespace_line = 0;
                    continue;
                }
            }

            /* check whitespace lines after the others because they use this to
             * determine if reading first non-whitespace char of the line */
            if (!isspace(c)) {
                whitespace_line = 0;

                /* check for code after comment end */
                if (!in_comment && found_comment_delim) {
                    partial_comment = 1;
                }
            }
        }

        if (whitespace_line)
            whitespace++;
        else if ((in_comment || found_comment_delim) && !partial_comment)
            /* test found_comment_delim in case the end of the line is the end
             * of the comment, in which case in_comment would be false */
            comments++;
    }

    printf(
        "Only comments: %d\nOnly whitespace: %d\nPreprocessor: "
        "%d\nCode: %d\nTotal: %d\n",
        comments, whitespace, preprocessor,
        total - comments - whitespace - preprocessor, total);

    fclose(in);
}

int main(int argc, char const* argv[]) {
    if (argc < 2) {
        puts("Provide a filename to read.");
        return 1;
    }
    count_lines_in(argv[1]);
    return 0;
}
