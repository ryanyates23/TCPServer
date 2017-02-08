#define __CL_ENABLE_EXCEPTIONS

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "cl.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#define WIDTH 1280
#define HEIGHT 720

//#define WIDTH 1024
//#define HEIGHT 576

//using namespace std;
using namespace cv;

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{
    //-------------BASIC INIT--------------------//
//    int argc = 51717;
//    char *argv = "localhost";
    int x, y;
    //int displayFrame = 0;
    int filterType = 3; //0 = none, 1 = 3x3mean, 2 = 3x3 median, 3 = 5x5 median, 4 = 3x3median 2 pass
    int enSobelEdge = 1;
    int enBinarise = 0;
    int enCombine = 0;
    int enCombineFiltering = 1;
    int width = WIDTH;
    int height = HEIGHT;
    
    int sum, max;
    float mean, thresh;
    float threshFactor = 0.25;
    float fps;
    
    
    
    
    //--------------OPENCV INIT---------------//
    uchar receiveBuffer[WIDTH*HEIGHT];
    uchar sendBuffer[WIDTH*HEIGHT];
    int vecSize = WIDTH*HEIGHT;
    std::vector<uchar> h_frame_in(vecSize);
    std::vector<uchar> h_frame_out(vecSize);
    
    //--------------OPENCL INIT--------------------//
    cl::Program program;
    cl::Context context;
    
    //This section sets up a variable to accept platform info, then gets it
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    //This selects the default platform
    cl::Platform platform = platforms.front();
    printf("Generated Platform\n");
    
    //This sets up a variable for the devices and then gets them
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    
    //This selects the default device
    cl::Device device = devices.front();
    printf("Established Device\n");
    
    std::cout << device.getInfo<CL_DEVICE_TYPE>();
    std::cout << CL_DEVICE_TYPE_GPU << std::endl;
    
    //This reads in the kernel file
    std::ifstream kernelFile("kernels.cl");
    std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
    printf("Read Kernel File\n");
    
    //This turns the string input into a source for kernels
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    
    //This creates an OpenCL context and a program from the sources
    context = cl::Context(device);
    printf("Created Context\n");
    program = cl::Program(context, sources);
    
    //This tries to build the kernels and outputs a build log if it fails miserably
    try {
        cl::Error err = program.build();
        printf("Built Program\n");
    }
    catch (cl::Error& e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            for (cl::Device dev : devices)
            {
                // Check the build status
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;
                
                // Get the build log
                std::string name = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                << buildlog << std::endl;
            }
        }
        else
        {
            throw e;
        }
        
    }
    
    cl::CommandQueue queue(context, device);
    
    cl::Buffer d_frame_in(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * h_frame_in.size(), h_frame_in.data());
    cl::Buffer d_original_frame(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uchar)*h_frame_in.size(), h_frame_in.data());
    cl::Buffer d_frame_out(context, CL_MEM_READ_WRITE, sizeof(uchar)*h_frame_out.size());
    
    cl::Kernel meanKernel(program, "MeanFilter");
    cl::Kernel medianKernel3(program, "MedianFilter3");
    cl::Kernel medianKernel5(program, "MedianFilter5");
    cl::Kernel sobelKernel(program, "SobelEdge");
    cl::Kernel binKernel(program, "Binarise");
    cl::Kernel combineKernel(program, "CombineImages");
    
    
    //------------TCP SERVER INIT--------------//
    int totalBytesRead;
    int totalBytesWritten;
    int sockfd, newsockfd, portno;
    socklen_t clilen;
    //char buffer[256];
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    if (argc < 2) {
        fprintf(stderr,"ERROR, no port provided\n");
        exit(1);
    }
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
             sizeof(serv_addr)) < 0)
        error("ERROR on binding");
    listen(sockfd,5);
    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd,
                       (struct sockaddr *) &cli_addr,
                       &clilen);
    if (newsockfd < 0)
        error("ERROR on accept");
    

    
    while(1)
    {
        
        //--------------------TCP RECEIVE----------------//
        bzero(receiveBuffer,vecSize);
        totalBytesRead = 0;
        while(totalBytesRead != vecSize)
        {
            n = (int)read(newsockfd,&receiveBuffer[totalBytesRead],vecSize - totalBytesRead);
            if (n < 0) error("ERROR reading from socket");
            
            totalBytesRead += n;
        }
        //std::cout << "Frame Received" << std::endl;
     
        //---------------DISPLAY FRAME------------------//

        //----------------PROCESSING----------------------//
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        
        for (y = 0; y < HEIGHT; y++)
        {
            for (x = 0; x < WIDTH-1; x++)
            {
                //std::cout << "Accessing Element: " << (y*WIDTH+x) << std::endl;
                h_frame_in[y*WIDTH + x] = receiveBuffer[y*WIDTH + x];
            }
        }
        
        
        
        queue.enqueueWriteBuffer(d_frame_in, CL_TRUE, 0, sizeof(uchar)*h_frame_in.size(), h_frame_in.data());
        queue.enqueueWriteBuffer(d_original_frame, CL_FALSE, 0, sizeof(uchar)*h_frame_in.size(), h_frame_in.data());
        
        if (filterType == 0)
        {
            queue.enqueueWriteBuffer(d_frame_out, CL_TRUE, 0, sizeof(uchar)*h_frame_in.size(), h_frame_in.data());
        }
        
        if (filterType == 1)
        {
            
            meanKernel.setArg(0, d_frame_in);
            meanKernel.setArg(1, d_frame_out);
            meanKernel.setArg(2, width);
            meanKernel.setArg(3, height);
            
            queue.enqueueNDRangeKernel(meanKernel, cl::NullRange, cl::NDRange(width, height));
            
            //queue.enqueueReadBuffer(d_frame_out, CL_TRUE, 0, sizeof(uchar)*h_frame_out.size(), h_frame_out.data());
            //queue.finish();
            //h_frame_in = h_frame_out;
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        if (filterType == 2)
        {
            
            medianKernel3.setArg(0, d_frame_in);
            medianKernel3.setArg(1, d_frame_out);
            medianKernel3.setArg(2, width);
            medianKernel3.setArg(3, height);
            
            queue.enqueueNDRangeKernel(medianKernel3, cl::NullRange, cl::NDRange(width, height));
            
            //queue.enqueueReadBuffer(d_frame_out, CL_TRUE, 0, sizeof(uchar)*h_frame_out.size(), h_frame_out.data());
            queue.finish();
            //h_frame_in = h_frame_out;
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        if(filterType == 3)
        {
            
            medianKernel5.setArg(0, d_frame_in);
            medianKernel5.setArg(1, d_frame_out);
            medianKernel5.setArg(2, width);
            medianKernel5.setArg(3, height);
            
            queue.enqueueNDRangeKernel(medianKernel5, cl::NullRange, cl::NDRange(width, height));
            queue.finish();
            //h_frame_in = h_frame_out;
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        if(filterType == 4)
        {
            medianKernel3.setArg(0, d_frame_in);
            medianKernel3.setArg(1, d_frame_out);
            medianKernel3.setArg(2, width);
            medianKernel3.setArg(3, height);
            
            queue.enqueueNDRangeKernel(medianKernel3, cl::NullRange, cl::NDRange(width, height));
            queue.finish();
            
            medianKernel3.setArg(0, d_frame_out);
            medianKernel3.setArg(1, d_frame_in);
            medianKernel3.setArg(2, width);
            medianKernel3.setArg(3, height);
            
            queue.enqueueNDRangeKernel(medianKernel3, cl::NullRange, cl::NDRange(width, height));
            queue.finish();
            
            queue.enqueueCopyBuffer(d_frame_out, d_frame_in, 0, 0, sizeof(uchar)*h_frame_out.size());
            queue.finish();
            
            
        }
        if (enCombine == 1 && enCombineFiltering == 1)
        {
            medianKernel5.setArg(0, d_original_frame);
            medianKernel5.setArg(1, d_original_frame);
            medianKernel5.setArg(2, width);
            medianKernel5.setArg(3, height);
            
            queue.enqueueNDRangeKernel(medianKernel5, cl::NullRange, cl::NDRange(width, height));
            
            queue.enqueueReadBuffer(d_frame_out, CL_TRUE, 0, sizeof(uchar)*h_frame_out.size(), h_frame_out.data());
            queue.finish();
        }
        if (enSobelEdge == 1)
        {
            
            sobelKernel.setArg(0, d_frame_out);
            sobelKernel.setArg(1, d_frame_in);
            sobelKernel.setArg(2, width);
            sobelKernel.setArg(3, height);
            
            queue.enqueueNDRangeKernel(sobelKernel, cl::NullRange, cl::NDRange(width, height));
            queue.finish();
        }
        else
        {
            h_frame_out = h_frame_in;
        }
        if(enBinarise == 1)
        {
            queue.enqueueReadBuffer(d_frame_in, CL_TRUE, 0, sizeof(uchar)*h_frame_in.size(), h_frame_in.data());
            
            sum = 0;
            max = 0;
            for (y = 0; y < height; y++)
            {
                for (x = 0; x < width; x++)
                {
                    sum += h_frame_in[y*width + x];
                    if(h_frame_in[y*width + x] > max) max = h_frame_in[y*width + x];
                }
            }
            
            mean = sum/(width*height);
            thresh = mean+((max-mean)*threshFactor);
            
            
            binKernel.setArg(0, d_frame_in);
            binKernel.setArg(1, d_frame_out);
            binKernel.setArg(2, width);
            binKernel.setArg(3, height);
            binKernel.setArg(4, (uchar)thresh);
            
            queue.enqueueNDRangeKernel(binKernel, cl::NullRange, cl::NDRange(width, height));
            queue.finish();
        }
        else
        {
            //h_frame_out = h_frame_in;
        }
        
        if (enCombine == 1 && enBinarise == 1)
        {
            
            combineKernel.setArg(0, d_frame_out);
            combineKernel.setArg(1, d_original_frame);
            combineKernel.setArg(2, d_frame_in);
            combineKernel.setArg(3, width);
            combineKernel.setArg(4, height);
            
            queue.enqueueNDRangeKernel(combineKernel, cl::NullRange, cl::NDRange(width, height));
            queue.finish();
        }
        else if (enCombine == 1 && enBinarise == 0)
        {
            
            combineKernel.setArg(0, d_frame_in);
            combineKernel.setArg(1, d_original_frame);
            combineKernel.setArg(2, d_frame_out);
            combineKernel.setArg(3, width);
            combineKernel.setArg(4, height);
            
            queue.enqueueNDRangeKernel(combineKernel, cl::NullRange, cl::NDRange(width, height));
            queue.finish();
        }
        
        if ((enBinarise == 0 && enCombine == 0) || (enBinarise == 1 && enCombine == 1))
        {
            queue.enqueueReadBuffer(d_frame_in, CL_TRUE, 0, sizeof(uchar)*h_frame_out.size(), h_frame_out.data());
        }
        else
        {
            queue.enqueueReadBuffer(d_frame_out, CL_TRUE, 0, sizeof(uchar)*h_frame_out.size(), h_frame_out.data());
        }
        
        
        
        for (y = 0; y < HEIGHT; y++)
        {
            for (x = 0; x < WIDTH-1; x++)
            {
                sendBuffer[y*WIDTH + x] = h_frame_out[y*WIDTH + x];
            }
        }
        
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
        fps = 1000000/duration;
        
        std::cout << fps << std::endl;

        //----------------TCP Send------------------------//
        totalBytesWritten = 0;
        while(totalBytesWritten != vecSize)
        {
            n = (int)write(newsockfd,&sendBuffer[totalBytesWritten],vecSize);
            //std::cout << "Bytes Written: " << n << std::endl;
            if (n < 0)
                error("ERROR writing to socket");
            totalBytesWritten += n;
        }

        
        //------------------------------------------------//
    }
    close(newsockfd);
    close(sockfd);
    while(1) x = 0;
    return 0;
}
