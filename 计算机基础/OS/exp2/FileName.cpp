#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>


const int BUFFER_SIZE = 6;
const int PRODUCE_TIMES = 12;
const int CONSUME_TIMES = 8;
const int STRING_LENGTH = 10;

std::queue<std::string> buffer;
std::mutex mtx;
std::condition_variable cv_full, cv_empty;

#include <random>

std::string generate_random_string(size_t length) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyz";
    std::string result;

    // 为每个线程生成独立的随机数生成器
    std::random_device rd;
    std::mt19937 generator(rd()); // 随机数引擎
    std::uniform_int_distribution<> distribution(0, sizeof(charset) - 2);

    for (size_t i = 0; i < length; ++i) {
        result += charset[distribution(generator)];
    }
    return result;
}


void print_buffer() {
    std::vector<std::string> temp_buffer;

    // 复制队列内容到临时向量
    
    temp_buffer = std::vector<std::string>(buffer.size());

    for (size_t i = 0; i < temp_buffer.size(); ++i) {
        temp_buffer[i] = buffer.front();
        buffer.pop();
    }

    std::cout << "当前缓冲区内容: ";
    for (const auto& item : temp_buffer) {
        std::cout << item << " ";
    }
    std::cout << std::endl;

    // 重新填充原始队列
    for (const auto& item : temp_buffer) {
        buffer.push(item);
    }
}


void producer(int producer_id) {
    for (int i = 0; i < PRODUCE_TIMES; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000)); // 随机睡眠
        std::string data = generate_random_string(STRING_LENGTH); // 产生随机数据

        std::unique_lock<std::mutex> lock(mtx);
        // 如果缓冲区满，则等待消费者消费
        while (buffer.size() == BUFFER_SIZE) {
            std::cout << "生产者序号： " << producer_id << " 生产了: " << data << " | 但是缓冲区已满。。。" << std::endl;
            cv_full.wait(lock); // 等待消费者通知
        }
        
        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);  // 转换为 time_t 格式
        std::tm local_time;                                                // 用于保存本地时间的结构体
        localtime_s(&local_time, &now_time);                               // 使用 localtime_s

        buffer.push(data);
        std::cout << "生产者序号： " << producer_id << " 生产了: " << data
            << " | 缓冲区大小: " << buffer.size() << "  | 当前时间: "
            << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")<<"  |  ";  // 使用put_time格式化输出


        print_buffer(); // 打印当前缓冲区内容

        cv_empty.notify_one(); // 通知消费者
    }
}

void consumer(int consumer_id) {
    for (int i = 0; i < CONSUME_TIMES; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 10000));

        std::unique_lock<std::mutex> lock(mtx);
        // 如果缓冲区空，则等待生产者生产
        while (buffer.empty()) {
            std::cout << "消费者序号： " << consumer_id << "  | 目前缓冲区已空。。。" << std::endl;
            cv_empty.wait(lock); // 等待生产者通知
        }
        
        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);  // 转换为 time_t 格式
        std::tm local_time;                                                // 用于保存本地时间的结构体
        localtime_s(&local_time, &now_time);                               // 使用 localtime_s

        std::string data = buffer.front();
        buffer.pop();
        std::cout << "消费者序号： " << consumer_id << " 消费了: " << data
            << " | 缓冲区大小: " << buffer.size() <<"  | 当前时间: "
            << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")<<"  |  ";

        print_buffer(); // 打印当前缓冲区内容

        cv_full.notify_one(); // 通知生产者
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));
    std::thread producers[2];
    std::thread consumers[3];

    // 创建生产者线程
    for (int i = 0; i < 2; ++i) {
        producers[i] = std::thread(producer, i + 1);
    }

    // 创建消费者线程
    for (int i = 0; i < 3; ++i) {
        consumers[i] = std::thread(consumer, i + 1);
    }

    // 等待所有线程完成
    for (int i = 0; i < 2; ++i) {
        producers[i].join();
    }

    for (int i = 0; i < 3; ++i) {
        consumers[i].join();
    }

    return 0;
}
