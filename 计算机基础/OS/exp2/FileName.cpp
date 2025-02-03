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

    // Ϊÿ���߳����ɶ����������������
    std::random_device rd;
    std::mt19937 generator(rd()); // ���������
    std::uniform_int_distribution<> distribution(0, sizeof(charset) - 2);

    for (size_t i = 0; i < length; ++i) {
        result += charset[distribution(generator)];
    }
    return result;
}


void print_buffer() {
    std::vector<std::string> temp_buffer;

    // ���ƶ������ݵ���ʱ����
    
    temp_buffer = std::vector<std::string>(buffer.size());

    for (size_t i = 0; i < temp_buffer.size(); ++i) {
        temp_buffer[i] = buffer.front();
        buffer.pop();
    }

    std::cout << "��ǰ����������: ";
    for (const auto& item : temp_buffer) {
        std::cout << item << " ";
    }
    std::cout << std::endl;

    // �������ԭʼ����
    for (const auto& item : temp_buffer) {
        buffer.push(item);
    }
}


void producer(int producer_id) {
    for (int i = 0; i < PRODUCE_TIMES; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000)); // ���˯��
        std::string data = generate_random_string(STRING_LENGTH); // �����������

        std::unique_lock<std::mutex> lock(mtx);
        // ���������������ȴ�����������
        while (buffer.size() == BUFFER_SIZE) {
            std::cout << "��������ţ� " << producer_id << " ������: " << data << " | ���ǻ���������������" << std::endl;
            cv_full.wait(lock); // �ȴ�������֪ͨ
        }
        
        // ��ȡ��ǰʱ��
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);  // ת��Ϊ time_t ��ʽ
        std::tm local_time;                                                // ���ڱ��汾��ʱ��Ľṹ��
        localtime_s(&local_time, &now_time);                               // ʹ�� localtime_s

        buffer.push(data);
        std::cout << "��������ţ� " << producer_id << " ������: " << data
            << " | ��������С: " << buffer.size() << "  | ��ǰʱ��: "
            << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")<<"  |  ";  // ʹ��put_time��ʽ�����


        print_buffer(); // ��ӡ��ǰ����������

        cv_empty.notify_one(); // ֪ͨ������
    }
}

void consumer(int consumer_id) {
    for (int i = 0; i < CONSUME_TIMES; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 10000));

        std::unique_lock<std::mutex> lock(mtx);
        // ����������գ���ȴ�����������
        while (buffer.empty()) {
            std::cout << "��������ţ� " << consumer_id << "  | Ŀǰ�������ѿա�����" << std::endl;
            cv_empty.wait(lock); // �ȴ�������֪ͨ
        }
        
        // ��ȡ��ǰʱ��
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);  // ת��Ϊ time_t ��ʽ
        std::tm local_time;                                                // ���ڱ��汾��ʱ��Ľṹ��
        localtime_s(&local_time, &now_time);                               // ʹ�� localtime_s

        std::string data = buffer.front();
        buffer.pop();
        std::cout << "��������ţ� " << consumer_id << " ������: " << data
            << " | ��������С: " << buffer.size() <<"  | ��ǰʱ��: "
            << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")<<"  |  ";

        print_buffer(); // ��ӡ��ǰ����������

        cv_full.notify_one(); // ֪ͨ������
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));
    std::thread producers[2];
    std::thread consumers[3];

    // �����������߳�
    for (int i = 0; i < 2; ++i) {
        producers[i] = std::thread(producer, i + 1);
    }

    // �����������߳�
    for (int i = 0; i < 3; ++i) {
        consumers[i] = std::thread(consumer, i + 1);
    }

    // �ȴ������߳����
    for (int i = 0; i < 2; ++i) {
        producers[i].join();
    }

    for (int i = 0; i < 3; ++i) {
        consumers[i].join();
    }

    return 0;
}
