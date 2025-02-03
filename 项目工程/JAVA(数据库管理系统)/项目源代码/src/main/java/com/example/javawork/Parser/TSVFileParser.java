package com.example.javawork.Parser;

import com.example.javawork.Database.DatabaseManager;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class TSVFileParser implements DataParser {
    private DatabaseManager databaseManager;

    public TSVFileParser(DatabaseManager dbManager) {
        this.databaseManager = dbManager;
    }

    @Override
    public void parserData(String filePath,String tableName) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            // 读取第一行作为列名
            String headerLine = reader.readLine();
            if (headerLine != null) {
                String[] columns = headerLine.split("\t"); // 假设使用制表符分隔列名
                // 创建表格或准备插入数据
                databaseManager.createTable(columns,tableName);

                // 逐行读取并插入数据
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] data = line.split("\t"); // 假设使用制表符分隔数据
                    databaseManager.insertData(columns, data,tableName);
                }
                System.out.println("Data parsed and stored successfully.");
            } else {
                System.out.println("File is empty.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
