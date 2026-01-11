#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include <unistd.h>
#include <termios.h>

typedef int32_t i32;
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

typedef i32 b32;

#define MIN(a, b) ((a) < (b) ? (a): (b))

typedef struct {
    u8* str;
    u64 size;
} string8;

#define STR8_LIT(s) (string8){ (u8*)(s), sizeof(s) - 1 }

typedef struct {
    struct termios orig_termios;

    u32 buf_capacity;
    u32 buf_size;
    u8* buf;
} term_context;

term_context* term_create(u32 capacity);
void term_quit(term_context* term);
u32 term_read(term_context* term, u8* keys, u32 capacity);
void term_write(term_context* term, string8 str);
void term_write_c(term_context* term, u8 c);
void term_flush(term_context* term);

typedef struct {
    u8 r, g, b;
} win_col;

typedef struct {
    u8 c;
    win_col fg_col;
    win_col bg_col;
} win_tile;

typedef struct {
    u32 width, height;
    win_tile* data;
} win_buf;

typedef struct {
    i32 x, y;
} vec2i;

typedef enum {
    DIR_UP,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT
} dir;

#define WIDTH 16
#define HEIGHT 16

b32 col_eq(win_col a, win_col b);
void set_col(term_context* term, win_col c, b32 fg);
b32 in_snake(vec2i* snake, u32 snake_size, vec2i pos);

int main(void) {
    srand(time(NULL));

    term_context* term = term_create(1 << 20);

    win_buf win = {
        WIDTH, HEIGHT,
        (win_tile*)malloc(sizeof(win_tile) * WIDTH * HEIGHT)
    };

    dir snake_dir = DIR_RIGHT;
    u32 snake_size = 1;
    vec2i* snake = (vec2i*)malloc(sizeof(vec2i) * WIDTH * HEIGHT);
    snake[0] = (vec2i){ WIDTH / 10, HEIGHT / 2 };

    vec2i apple = {
        rand() % WIDTH,
        rand() % HEIGHT,
    };

    while (1) {
        u8 input = 0;
        read(STDIN_FILENO, &input, 1);

        if (input == 'q') { break; }

        // Update
        switch (input) {
            case 'w': { snake_dir = DIR_UP; } break;
            case 'a': { snake_dir = DIR_LEFT; } break;
            case 's': { snake_dir = DIR_DOWN; } break;
            case 'd': { snake_dir = DIR_RIGHT; } break;
        }

        vec2i head = snake[snake_size - 1];

        switch (snake_dir) {
            case DIR_UP: { head.y--; } break;
            case DIR_DOWN: { head.y++; } break;
            case DIR_LEFT: { head.x--; } break;
            case DIR_RIGHT: { head.x++; } break;
        }

        if (head.x < 0 || head.x >= WIDTH || head.y < 0 || head.y >= HEIGHT) {
            break;
        }
        if (in_snake(snake, snake_size, head)) {
            break;
        }

        if (head.x == apple.x && head.y == apple.y) {
            do {
                apple.x = rand() % WIDTH;
                apple.y = rand() % HEIGHT;
            } while(in_snake(snake, snake_size, apple) || (head.x == apple.x && head.y == apple.y));

            snake[snake_size++] = head;
        } else {
            for (u32 i = 0; i < snake_size - 1; i++) {
                snake[i] = snake[i + 1];
            }
            snake[snake_size-1] = head;
        }

        // Draw
        for (u32 i = 0; i < WIDTH * HEIGHT; i++) {
            win.data[i].c = '.';
            win.data[i].fg_col = (win_col){ 255, 255, 255 };
            win.data[i].bg_col = (win_col){ 0, 0, 0 };
        }

        win.data[apple.x + apple.y * win.width] = (win_tile){
            .c = 'A',
            .fg_col = (win_col){ 255, 0, 0 },
            .bg_col = (win_col){ 0, 0, 0 },
        };

        for (u32 i = 0; i < snake_size; i++) {
            vec2i seg = snake[i];

            win.data[seg.x + seg.y * win.width] = (win_tile) {
                .c = '#',
                .fg_col = (win_col){ 0, 255, 0 },
                .bg_col = (win_col){ 0, 0, 0 },
            };
        }

        term_write(term, STR8_LIT("\x1b[2J\x1b[H"));
        
        set_col(term, win.data[0].fg_col, 1);
        set_col(term, win.data[0].bg_col, 0);

        win_col prev_fg = win.data[0].fg_col;
        win_col prev_bg = win.data[0].bg_col;

        for (u32 y = 0; y < HEIGHT; y++) {
            for (u32 x = 0; x < WIDTH; x++) {
                u32 index = x + y * WIDTH;

                if (!col_eq(prev_fg, win.data[index].fg_col)) {
                    set_col(term, win.data[index].fg_col, true);
                    prev_fg = win.data[index].fg_col;
                }
                if (!col_eq(prev_bg, win.data[index].bg_col)) {
                    set_col(term, win.data[index].bg_col, true);
                    prev_bg = win.data[index].bg_col;
                }

                term_write_c(term, win.data[index].c);
            }
            term_write(term, STR8_LIT("\x1b[1E"));
        }

        term_flush(term);

        usleep(200 * 1000);
    }

    term_quit(term);

    return 0;
}

b32 col_eq(win_col a, win_col b) {
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

void set_col(term_context* term, win_col c, b32 fg) {
    u8 chars[21] = { 0 };

    u32 size = snprintf(
        (char*)chars, sizeof(chars),
        "\x1b[%d8;2;%d;%d;%dm",
        fg ? 3 : 4, c.r, c.g, c.b
    );

    term_write(term, (string8){ chars, size });
}

b32 in_snake(vec2i* snake, u32 snake_size, vec2i pos) {
    for (u32 i = 0; i < snake_size; i++) {
        if (snake[i].x == pos.x && snake[i].y == pos.y) {
            return true;
        }
    }

    return false;
}

term_context* term_create(u32 capacity) {
    term_context* term = (term_context*)malloc(sizeof(term_context));

    tcgetattr(STDIN_FILENO, &term->orig_termios);

    struct termios raw = term->orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    term->buf_capacity = capacity;
    term->buf_size = 0;
    term->buf = (u8*)malloc(capacity);
    memset(term->buf, 0, capacity);

    string8 to_write = STR8_LIT("\x1b[?1049h\x1b[2J");
    write(STDOUT_FILENO, to_write.str, to_write.size);

    return term;
}

void term_quit(term_context* term) {
    string8 to_write = STR8_LIT("\x1b[?1049l");
    write(STDOUT_FILENO, to_write.str, to_write.size);

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &term->orig_termios);

    free(term);
}

u32 term_read(term_context* term, u8* keys, u32 capacity) {
    return read(STDIN_FILENO, keys, capacity);
}

void term_write(term_context* term, string8 str) {
    while (str.size > 0) {
        u32 to_write = MIN(term->buf_capacity - term->buf_size, str.size);

        memcpy(term->buf + term->buf_size, str.str, to_write);
        term->buf_size += to_write;

        str.str += to_write;
        str.size -= to_write;

        if (term->buf_size >= term->buf_capacity) {
            term_flush(term);
        }
    }
}

void term_write_c(term_context* term, u8 c) {
    term->buf[term->buf_size++] = c;

    if (term->buf_size >= term->buf_capacity) {
        term_flush(term);
    }
}

void term_flush(term_context* term) {
    write(STDOUT_FILENO, term->buf, term->buf_size);
    term->buf_size = 0;
}

