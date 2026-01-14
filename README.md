An extension of @magicalbat's machine learning library

https://garden.tanushswaminathan.com/writings/science/Genomics-ML-Library-Speedrun

mlfromscratch/
├── genoml/                  # Main Python package
│   ├── sequence/            # Encoding (one-hot, k-mer, positional)
│   ├── io/                  # FASTA/FASTQ readers/writers
│   ├── utils/               # Sequence utilities & alignment
│   ├── layers/              # Neural network layers (Conv1D, Dense, Attention, PWM)
│   └── losses/              # Loss functions & metrics
├── machine-learning/        # Original C ML implementation
│   ├── main.c               # MNIST training demo
│   └── arena.c/prng.c       # Memory & RNG utilities
├── examples/                # Runnable example scripts
├── setup.py                 # Package installation
└── requirements.txt         # Dependencies (numpy)

--> import genoml
