#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace py = pybind11;

struct HorizonObservation {
    bool valid;
    int64_t timestamp_sec;
    int64_t timestamp_nsec;
    double roll_rad;
    double pitch_rad;
    double roll_var;
    double pitch_var;  
};

class ImageProcessor {
public:
    ImageProcessor(py::array_t<double> mtx, py::array_t<double> dist,
                   int img_width, int img_height, double camera_height) : 
                   width(img_width), height(img_height), cam_height(camera_height) {
        // Convert numpy arrays to cv::Mat
        this->mtx = cv::Mat(3, 3, CV_64F, (double*)mtx.request().ptr).clone();
        this->dist = cv::Mat(1, 5, CV_64F, (double*)dist.request().ptr).clone();

        this->new_mtx = cv::getOptimalNewCameraMatrix(this->mtx, this->dist, cv::Size(width, height), 1);

        this->lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);

        min_line_length = width / 3; 
        vertical_buffer = height / 10;
        }
    
    HorizonObservation lsd_image_callback(py::array_t<unsigned char> image, int64_t sec, int64_t nsec, bool debug) {
        py::buffer_info buf = image.request();
        cv::Mat frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

        // TODO: All the image logic here:
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

        // undistort the image
        cv::Mat undistorted;
        cv::undistort(gray_frame, undistorted, mtx, dist, new_mtx);

        std::vector<cv::Vec4f> lines;
        lsd->detect(undistorted, lines);

        std::vector<double> lengths;
        for (const auto& line : lines) {
            lengths.push_back(compute_line_length(line));
        }

        // Only keep lines 1) longer than 1/3 img_width 
        //                 2) not vertical
        //                 3) Not near the top or bottom 1/10 of the image (undistort creates false lines)
        // std::vector<cv::Vec4f> filtered_lines;
        // for (size_t i = 0; i < lines.size(); ++i) {
        //     if (lengths[i] >= min_line_length && 
        //             std::abs(lines[i][1] - lines[i][3]) <= 2 &&
        //             lines[i][1] > vertical_buffer && lines[i][3] > vertical_buffer &&
        //             lines[i][1] < height - vertical_buffer && lines[i][3] < height - vertical_buffer) {
        //         filtered_lines.push_back(lines[i]);
        //     }
        // }
        // 1. Filter with the same logic as Python
        std::vector<cv::Vec4f> filtered_lines;
        for (size_t i = 0; i < lines.size(); ++i) {
            float dx = std::abs(lines[i][0] - lines[i][2]);
            float dy = std::abs(lines[i][1] - lines[i][3]);
            
            // Logic: Must have width (dx > 2) and be within the vertical buffers
            if (lengths[i] >= min_line_length && 
                dx > 2 && 
                lines[i][1] > vertical_buffer && lines[i][3] > vertical_buffer &&
                lines[i][1] < height - vertical_buffer && lines[i][3] < height - vertical_buffer) {
                filtered_lines.push_back(lines[i]);
            }
        }

        // 2. Sort by length (Longest first) to match Python's intent
        std::sort(filtered_lines.begin(), filtered_lines.end(), 
            [this](const cv::Vec4f& a, const cv::Vec4f& b) {
                return compute_line_length(a) > compute_line_length(b);
            });

        HorizonObservation obs;

        if (filtered_lines.empty()) {
            obs.valid = false;
            obs.timestamp_sec = sec;
            obs.timestamp_nsec = nsec;
            obs.roll_rad = 0.0;
            obs.pitch_rad = 0.0;
            obs.roll_var = 0.0;
            obs.pitch_var = 0.0;
            return obs;
        }
        cv::Vec4f line = filtered_lines[0];
        // x1, y1, x2, y2 = line[0]
        if (line[0] > line[2]) {
            std::swap(line[0], line[2]);
            std::swap(line[1], line[3]);
        }
        double angle = std::atan2(line[3] - line[1], line[2] - line[0])*-1;
        
        double yL = line[1] + line[0] * std::tan(angle);
        double yR = line[3] - (width - line[2]) * std::tan(angle);

        double y_intercept = (yL + yR) / 2.0;
        double delta_y = y_intercept - new_mtx.at<double>(1, 2); // principal point y
        double psi = std::atan2(delta_y, new_mtx.at<double>(1, 1)); // focal length in y

        double theta = -1 * psi + M_PI/2.0 - std::asin(Re / (Re + cam_height));
        
        // roll = est_angle
        // pitch = theta
        obs.valid = true;
        obs.timestamp_sec = sec;
        obs.timestamp_nsec = nsec;
        obs.roll_rad = angle;
        obs.pitch_rad = theta;
        obs.roll_var = 0.0026179939; // 0.15 deg in radians
        obs.pitch_var = 0.0026179939; // 0.15 deg in radians

        return obs;

        }
private:
    cv::Mat mtx, dist, new_mtx;
    int width, height;
    double cam_height;
    cv::Ptr<cv::LineSegmentDetector> lsd;

    int min_line_length;
    int vertical_buffer;

    int Re = 6378137; // in meters


    double compute_line_length(const cv::Vec4f& line) {
        return std::sqrt(std::pow(line[2] - line[0], 2) + std::pow(line[3] - line[1], 2));
    }
};

PYBIND11_MODULE(vision_cpp, m) {
py::class_<HorizonObservation>(m, "HorizonObservation")
        .def_readonly("valid", &HorizonObservation::valid)
        .def_readonly("timestamp_sec", &HorizonObservation::timestamp_sec)
        .def_readonly("timestamp_nsec", &HorizonObservation::timestamp_nsec)
        .def_readonly("roll_rad", &HorizonObservation::roll_rad)
        .def_readonly("pitch_rad", &HorizonObservation::pitch_rad)
        .def_readonly("roll_var", &HorizonObservation::roll_var)
        .def_readonly("pitch_var", &HorizonObservation::pitch_var);
        
    py::class_<ImageProcessor>(m, "ImageProcessor")
        .def(py::init<py::array_t<double>, py::array_t<double>, int, int, double>())
        .def("lsd_image_callback", &ImageProcessor::lsd_image_callback, 
             py::arg("image"), py::arg("sec")=0, py::arg("nsec")=0, py::arg("debug")=false);
}