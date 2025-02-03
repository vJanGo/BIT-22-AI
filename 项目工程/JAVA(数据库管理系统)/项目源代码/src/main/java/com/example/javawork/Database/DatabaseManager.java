package com.example.javawork.Database;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class DatabaseManager {
    private Connection connection;
    private String dbName = "identifier.sqlite"; // 数据库名称

    public static List<List<String>> matchingData = new ArrayList<>();


    public DatabaseManager() {
        try {
            // 注册SQLite JDBC驱动
            Class.forName("org.sqlite.JDBC");
            // 连接SQLite数据库
            connection = DriverManager.getConnection("jdbc:sqlite:" + dbName);
            System.out.println("Connected to SQLite database.");
        } catch (ClassNotFoundException | SQLException e) {
            e.printStackTrace();
        }
    }

    public boolean isTableExists(String tableName) {
        try {
            // 查询表是否存在
            PreparedStatement pstmt = connection.prepareStatement(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?");
            pstmt.setString(1, tableName);
            return pstmt.executeQuery().next();
        } catch (SQLException e) {
            e.printStackTrace();
            return false;
        }
    }

    // 创建表格方法，传入列名数组
    public void createTable(String[] columns, String tableName) {
        try {
            if (!isTableExists(tableName)) {
                StringBuilder createTableSQL = new StringBuilder("CREATE TABLE IF NOT EXISTS " + tableName + "(");
                for (String column : columns) {
                    // 引用带空格的列名
                    if (column.contains(" ")) {
                        createTableSQL.append("`").append(column).append("` TEXT, ");
                    } else {
                        createTableSQL.append(column).append(" TEXT, ");
                    }
                }
                createTableSQL.deleteCharAt(createTableSQL.length() - 2); // 移除最后一个逗号和空格
                createTableSQL.append(")");

                PreparedStatement statement = connection.prepareStatement(createTableSQL.toString());
                statement.executeUpdate();
                System.out.println("Table created successfully.");
            } else {
                System.out.println("Table already exists.");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }


    // 插入数据方法，传入列名数组和数据数组
    public void insertData(String[] columns, String[] data, String tableName) {
        try {
            StringBuilder insertQuery = new StringBuilder("INSERT INTO " + tableName + "(");

            // 构建列名部分，并对带空格的列名进行引用
            for (int i = 0; i < columns.length; i++) {
                if (columns[i].contains(" ")) {
                    insertQuery.append("`").append(columns[i]).append("`");
                } else {
                    insertQuery.append(columns[i]);
                }

                if (i != columns.length - 1) {
                    insertQuery.append(", ");
                }
            }

            // 构建值部分的占位符
            insertQuery.append(") VALUES (");
            for (int i = 0; i < columns.length; i++) {
                insertQuery.append("?");
                if (i != columns.length - 1) {
                    insertQuery.append(", ");
                }
            }
            insertQuery.append(")");

            PreparedStatement pstmt = connection.prepareStatement(insertQuery.toString());

            // 设置参数
            for (int i = 0; i < data.length; i++) {
                pstmt.setString(i + 1, data[i]);
            }

            pstmt.executeUpdate();
            System.out.println("Data inserted successfully.");
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }


    public void closeConnection() {
        try {
            if (connection != null) {
                connection.close();
                System.out.println("Connection closed.");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public int getRowCount(String tableName) {
        int rowCount = 0;
        try {
            PreparedStatement pstmt = connection.prepareStatement("SELECT COUNT(*) FROM " + tableName);
            ResultSet resultSet = pstmt.executeQuery();
            if (resultSet.next()) {
                rowCount = resultSet.getInt(1);
            }
            resultSet.close();
            pstmt.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return rowCount;
    }

    public int columnCount;

    public List<List<String>> GetData(String tableName, int currentPage, int rowsPerPage) {
        List<List<String>> pageData = new ArrayList<>();
        try {
            PreparedStatement pstmt = connection.prepareStatement(
                    "SELECT * FROM " + tableName + " LIMIT ? OFFSET ?");
            pstmt.setInt(1, rowsPerPage);
            pstmt.setInt(2, (currentPage - 1) * rowsPerPage);
            ResultSet resultSet = pstmt.executeQuery();

            int columnCount = resultSet.getMetaData().getColumnCount();

            while (resultSet.next()) {
                List<String> rowData = new ArrayList<>();
                for (int i = 1; i <= columnCount; i++) {
                    rowData.add(resultSet.getString(i));
                }
                pageData.add(rowData);
            }

            resultSet.close();
            pstmt.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return pageData;
    }

    public void printPageData(List<List<String>> pageData) {
        for (List<String> rowData : pageData) {
            for (String data : rowData) {
                System.out.print(data + "\t");
            }
            System.out.println();
        }
    }

    // 获取表的列名列表
    public List<String> getTableColumnNames(String tableName) throws SQLException {
        List<String> columnNames = new ArrayList<>();

        // 获取数据库的元数据
        DatabaseMetaData metaData = connection.getMetaData();
        try (ResultSet rs = metaData.getColumns(null, null, tableName, null)) {
            while (rs.next()) {
                String columnName = rs.getString("COLUMN_NAME");
                columnNames.add(columnName);
            }
        }
        columnCount = columnNames.size();
        return columnNames;
    }

    public List<String> getDatabaseNames() throws SQLException {
        List<String> databaseNames = new ArrayList<>();

        try {
            DatabaseMetaData metaData = connection.getMetaData();
            ResultSet resultSet = metaData.getTables(null, null, "%", null);

            while (resultSet.next()) {
                String tableName = resultSet.getString("TABLE_NAME");
                databaseNames.add(tableName);
            }

            resultSet.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }


        return databaseNames;
    }

    public void searchDataByColumn(String[] keywords, String columnName, String tableName,boolean exactMatch) {

        try {
            StringBuilder queryBuilder = new StringBuilder("SELECT * FROM "+tableName+" WHERE ");

            if (exactMatch) {
                // 添加非精准匹配条件
                for (int i = 0; i < keywords.length; i++) {
                    queryBuilder.append(columnName).append(" LIKE ?");
                    if (i < keywords.length - 1) {
                        queryBuilder.append(" OR ");
                    }
                }
            } else {
                // 添加精准匹配条件，只要列中包含所有关键字就匹配成功
                queryBuilder.append("1=1");
                for (int i = 0; i < keywords.length; i++) {
                    queryBuilder.append(" AND ").append(columnName).append(" LIKE ?");
                }
            }

            PreparedStatement pstmt = connection.prepareStatement(queryBuilder.toString());

            // 设置参数，将通配符和关键词组合
            for (int i = 0; i < keywords.length; i++) {
                if (exactMatch) {
                    pstmt.setString(i + 1, keywords[i]);
                } else {
                    pstmt.setString(i + 1, "%" + keywords[i] + "%");
                }
            }

            ResultSet resultSet = pstmt.executeQuery();

            int columnCount = resultSet.getMetaData().getColumnCount();

            // 将匹配的数据添加到列表中
            while (resultSet.next()) {
                List<String> rowData = new ArrayList<>();
                for (int i = 1; i <= columnCount; i++) {
                    rowData.add(resultSet.getString(i));
                }
                matchingData.add(rowData);
            }

            resultSet.close();
            pstmt.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public int getSearchDataRowCount() {
        return matchingData.size();
    }

    public List<List<String>> GetSearchData(int pageIndex,int pageSize){
        List<List<String>> pageData = new ArrayList<>();
        int StartIndex = (pageIndex-1) * pageSize;
        int EndIndex = Math.min(StartIndex + pageSize, matchingData.size());
        for (int i = StartIndex; i < EndIndex; i++) {
            pageData.add(new ArrayList<>(matchingData.get(i)));
        }
        //matchingData.clear();//初始化方便下次调用
        return pageData;
    }






    /*public static void main(String[] args) throws SQLException {
        // 测试数据库管理类
        DatabaseManager dbManager = new DatabaseManager();
        String[] key = {"A","1"};
        String column = "Entry";
        List<List<String>> res = dbManager.searchDataByColumn(key,column,"aaa",true);
        dbManager.closeConnection();
    }*/

}