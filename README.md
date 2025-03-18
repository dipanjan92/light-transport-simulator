# Light Transport Simulator

This is a **work-in-progress** physically based light transport simulator using [**Taichi Lang**](https://www.taichi-lang.org/). The project heavily draws from **PBRT** ([Pharr, Jakob, & Humphreys, 2023](https://pbr-book.org/4ed/contents)) and aims to implement foundational light transport algorithms.

## Status and Scope

- While **Taichi** supports GPU acceleration, all functionalities have been tested **only on CPU** at this stage.
- The current implementation is focused on fundamental rendering techniques.
- The project uses a simple [PBRT v4 scene parser](./pbrt) (work in progress) for loading scenes.
- Check the [examples](./examples) directory to see how the rendering pipeline works.

## References

This project draws from established works in rendering and physically based light transport:

- **PBRT**: Pharr, M., Jakob, W., & Humphreys, G., 2023. *Physically Based Rendering: From Theory to Implementation.* 4th ed. [Available here](https://pbr-book.org/4ed/contents).
- **Real-Time Rendering**: Akenine-Möller, T., Haines, E., & Hoffman, N., 2018. *Real-Time Rendering.* 4th ed. CRC Press.
- **Mitsuba 3**: Jakob, W., Speierer, S., Roussel, N., et al., 2022. *Mitsuba 3 Renderer (v3.1.1).* [Mitsuba Website](https://mitsuba-renderer.org).
- **Monte Carlo Methods**: Veach, E., 1997. *Robust Monte Carlo Methods for Light Transport Simulation.* Ph.D. thesis, Stanford University.
- **Taichi Three**: Taichi Developers, 2024. *taichi_three.* [GitHub Repository](https://github.com/taichi-dev/taichi_three).
- **Rendering Equation**: Raviramamoorthi, 2020. *Online Computer Graphics II: Rendering: Theory: Rendering Equation.* [YouTube Video](https://youtu.be/fyIA5h2UYGk).
- **MIT Introduction to Computer Graphics**: Solomon, J., 2020. *6.837: Introduction to Computer Graphics (Fall 2020).* [YouTube Playlist](https://www.youtube.com/watch?v=-LqUu61oRdk&list=PLQ3UicqQtfNuBjzJ-KEWmG1yjiRMXYKhh&pp=iAQB).
- **Ray Tracing Course (TU Wien)**: Two Minute Papers, 2015. *TU Wien Rendering / Ray Tracing Course.* [YouTube Playlist](https://www.youtube.com/watch?v=pjc1QAI6zS0&list=PLujxSBD-JXgnGmsn7gEyN28P1DnRZG7qi&pp=iAQB).
