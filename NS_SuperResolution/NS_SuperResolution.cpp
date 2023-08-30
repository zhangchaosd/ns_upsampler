#include <iostream>
#include "Audio.h"
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"

using namespace ov::preprocess;

int getNSDeviceId() {
    std::vector<int> availableCameras;
    cv::VideoCapture cap;
    for (int i = 0; i < 10; ++i) {
        if (cap.open(i)) {
            std::cout << "Found capture device: " << i << std::endl;
            availableCameras.push_back(i);
            cap.release();
        }
        else {
            break;
        }
    }

    if (availableCameras.empty()) {
        std::cout << "No capture device found。" << std::endl;
        return -1;
    }

    std::cout << "Please select a capture device: ";
    int choice;
    std::cin >> choice;

    if (std::find(availableCameras.begin(), availableCameras.end(), choice) == availableCameras.end()) {
        std::cout << "Invalid number" << std::endl;
        return -1;
    }
    return choice;
}

std::string getDevice(ov::Core& core) {
    std::vector<std::string> devices = core.get_available_devices();
    if (devices.empty()) {
        std::cout << "No available device found" << std::endl;
        return "";
    }
    std::string selected_device = "CPU";
    std::cout << "Found devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++) {
        std::cout << i << ": " << devices[i] << std::endl;
    }
    std::cout << "Please select a device to run the model:";
    int choice;
    std::cin >> choice;

    if (choice < 0 || choice >= devices.size()) {
        std::cout << "Invalid number" << std::endl;
        return "";
    }
    return devices[choice];
}

int main(int argc, char* argv[])
{
    std::cout<<"!!!!!  Press Q to exit while streaming  !!!!!!" << std::endl;
    int deviceID = getNSDeviceId();
    if (deviceID == -1) {
        return 0;
    }
    //AudioStart();
    std::cout << ov::get_openvino_version() << std::endl;
    const std::string model_path = "SRNet.onnx";

    ov::Core core;
    std::string device = getDevice(core);


    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    std::string input_tensor_name = model->input().get_any_name();
    std::string output_tensor_name = model->output().get_any_name();
    ov::CompiledModel compiled_model = core.compile_model(model, device);
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    const int total_size = 1080 * 1920 * 3;
    std::shared_ptr<uint8_t> image_data(new uint8_t[total_size], std::default_delete<uint8_t[]>());

    cv::VideoCapture cap;

    cap.open(deviceID);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 60);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR!!Unable to open camera\n";
        return -1;
    }

    cv::Mat img;
    while (true)
    {
        cap >> img;

        cv::namedWindow("NS_SuperResolution", cv::WINDOW_NORMAL);
        if (img.empty() == false)
        {
            ov::Tensor input_tensor{ ov::element::u8, {1080,1920,3}, img.data};
            infer_request.set_tensor(input_tensor_name, input_tensor);
            infer_request.infer();
            ov::Tensor output = infer_request.get_tensor(output_tensor_name);
            cv::Mat img_hr(2160, 3840, CV_8UC3, (uint8_t*)output.data());
            cv::imshow("example", img_hr);
        }
        else {
            std::cout << "no pic" << std::endl;
        }

        int  key = cv::waitKey(10);
        if (key == int('q') || key == int('Q'))
        {
            break;
        }

    }
    cap.release();
    cv::destroyAllWindows();
   // AudioStop();
    return 0;
}
