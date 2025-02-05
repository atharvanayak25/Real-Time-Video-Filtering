#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Hardcode the image path
    std::string imagePath = "C:/Users/Atharva/Pictures/Screenshots/Screenshot 2023-08-21 211928.png";

    // Read the image file
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Check if the image was successfully loaded
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return -1;
    }

    // Create a window to display the image
    std::string windowName = "Image Display";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Display the image in the created window
    cv::imshow(windowName, image);

    std::cout << "Press 'q' to quit or try other keys!" << std::endl;

    // Enter a loop to wait for keypress
    while (true) {
        int key = cv::waitKey(10); // Wait for 10 ms

        if (key == 'q') {
            std::cout << "Exiting program..." << std::endl;
            break; // Exit the loop if 'q' is pressed
        } else if (key == 's') {
            // Save the displayed image if 's' is pressed
            cv::imwrite("saved_image.jpg", image);
            std::cout << "Image saved as 'saved_image.jpg'" << std::endl;
        } else if (key > 0) {
            std::cout << "You pressed key: " << static_cast<char>(key) << std::endl;
        }
    }

    // Destroy the window before exiting
    cv::destroyWindow(windowName);

    return 0;
}
