Piano Next Note Generation
==================================

In this project, we trained an LSTM to predict the next note/s of a given input piano midi file.

### Project Highlights:
- Main reference used is the [Tensorflow](https://www.tensorflow.org/overview) guide to audio data.
- The dataset used (maestro dataset) is a publicly available collection of piano midi files.
- The training objective is to build a model that would minimize the overall loss composed of pitch, step, and duration losses.
- The prediction can vary depending on the desired randomness/entropy and quantity.

### Hardware:
- Nvidia GeForce RTX 2060 Mobile GPU
- Ryzen 7 4800H CPU

### Operating System:
- Windows 11

### Try the code:
Place the midi file inside test/inputs folder, navigate inside the src folder and issue the following command:
(_Suggested entropy values = 1.0-3.0)
```cmd
python predict.py <your_video_filename> <entropy_value_in_float> <num_notes_in_int>
```

### Credits/Citations:

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jozefowicz, R., Jia, Y., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Schuster, M., Monga, R., Moore, S., Murray, D., Olah, C., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., & Zheng, X. (2015). TensorFlow, Large-scale machine learning on heterogeneous systems [Computer software]. https://doi.org/10.5281/zenodo.4724125
