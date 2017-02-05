__kernel void SobelEdge(__global uchar* in, __global uchar* out, const int width, const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int offset = y*width + x;
    uchar data[9];
    float dx = 0;
    float dy = 0;
    float dout = 0;
    
    if(x > 1 && x < width-2 && y > 1 && y < height-2)
    {
        data[0] = in[offset-width-1];
        data[1] = in[offset-width];
        data[2] = in[offset-width+1];
        data[3] = in[offset-1];
        data[4] = in[offset];
        data[5] = in[offset+1];
        data[6] = in[offset+width-1];
        data[7] = in[offset+width];
        data[8] = in[offset+width+1];
        
        dx = -data[0] + data[2] - 2*data[3] + 2*data[5] - data[6] + data[8];
        dy = -data[0] - 2*data[1] - data[2] + data[6] + 2*data[7] + data[8];
        
        dout = sqrt(dx*dx + dy*dy);
        
        if(dout > 255) out[offset] = 255;
        else if(dout < 0) out[offset] = 0;
        else out[offset] = convert_uchar(dout);
        
        //out[offset] = in[offset];
    }
    else out[offset] = 0;
    

}

__kernel void MeanFilter(__global uchar* in, __global uchar* out, const int width, const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int offset = y*width + x;
    float mean = 0;
    float sum = 0;
    
    if(x > 1 && x < width-2 && y > 1 && y < height-2)
    {
        sum = in[offset-width-1] + in[offset-width] + in[offset-width+1] + in[offset-1] + in[offset] + in[offset+1] + in[offset+width-1] + in[offset+width] + in[offset+width+1];
        mean = sum/9;
        
        out[offset] = convert_uchar(mean);
        
    }
    else
    {
        out[offset] = in[offset];
    }
}

__kernel void MedianFilter3(__global uchar* in, __global uchar* out, const int width, const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int offset = y*width + x;
    uchar data[9];
    uchar temp;
    int swapped = 1;
    int i;
    
    if(x > 1 && x < width-2 && y > 1 && y < height-2)
    {
        data[0] = in[offset-width-1];
        data[1] = in[offset-width];
        data[2] = in[offset-width+1];
        data[3] = in[offset-1];
        data[4] = in[offset];
        data[5] = in[offset+1];
        data[6] = in[offset+width-1];
        data[7] = in[offset+width];
        data[8] = in[offset+width+1];
        
        while(swapped == 1)
        {
            swapped = 0;
            for(i=0;i<7;i++)
            {
                if(data[i] > data[i+1])
                {
                    temp = data[i];
                    data[i] = data[i+1];
                    data[i+1] = temp;
                    swapped = 1;
                }
            }
        }
        out[offset] = data[4];
    }
    else
    {
        out[offset] = in[offset];
    }
    
}

__kernel void MedianFilter5(__global uchar* in, __global uchar* out, const int width, const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int offset = y*width + x;
    uchar data[25];
    uchar temp;
    int swapped = 1;
    int i;
    
    if(x > 1 && x < width-2 && y > 1 && y < height-2)
    {
        data[0] = in[offset-2*width-2];
        data[1] = in[offset-2*width-1];
        data[2] = in[offset-2*width];
        data[3] = in[offset-2*width+1];
        data[4] = in[offset-2*width+2];
        data[5] = in[offset-width-2];
        data[6] = in[offset-width-1];
        data[7] = in[offset-width];
        data[8] = in[offset-width+1];
        data[9] = in[offset-width+2];
        data[10] = in[offset-2];
        data[11] = in[offset-1];
        data[12] = in[offset];
        data[13] = in[offset+1];
        data[14] = in[offset+2];
        data[15] = in[offset+width-2];
        data[16] = in[offset+width-1];
        data[17] = in[offset+width];
        data[18] = in[offset+width+1];
        data[19] = in[offset+width+2];
        data[20] = in[offset+2*width-2];
        data[21] = in[offset+2*width-1];
        data[22] = in[offset+2*width];
        data[23] = in[offset+2*width+1];
        data[24] = in[offset+2*width+2];
        
        while(swapped == 1)
        {
            swapped = 0;
            for(i=0;i<23;i++)
            {
                if(data[i] > data[i+1])
                {
                    temp = data[i];
                    data[i] = data[i+1];
                    data[i+1] = temp;
                    swapped = 1;
                }
            }
        }
        out[offset] = data[12];
    }
    else
    {
        out[offset] = in[offset];
    }
}

__kernel void Binarise(__global uchar* in, __global uchar* out, const int width, const int height, const uchar thresh)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int offset = y*width + x;
    
    if(in[offset] > thresh)
    {
        //out[offset] = in[offset];
        out[offset] = 255;
    }
    else
    {
        out[offset] = 0;
    }
    
}

__kernel void CombineImages(__global uchar* in1, __global uchar* in2, global uchar* out, const int width, const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int offset = y*width + x;
    int temp;

    
    temp = in1[offset] + in2[offset];
    if (temp > 255) out[offset] = 255; else out[offset] = temp;
}












