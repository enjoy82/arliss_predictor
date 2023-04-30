#pragma once
#include <vector>

struct Xyxy{
    int x1;
    int y1;
    int x2;
    int y2;

    Xyxy(int x1, int y1, int x2, int y2){
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
    }

    std::vector<float> normalized(int height, int width){        
        return std::vector<float> {
            static_cast<float>(this->x1) / static_cast<float>(width),
            static_cast<float>(this->y1) / static_cast<float>(height),
            static_cast<float>(this->x2) / static_cast<float>(width),
            static_cast<float>(this->y2) / static_cast<float>(height),
            };
    }
};
