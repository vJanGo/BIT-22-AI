package com.example.javawork.FileSelector;

import com.example.javawork.Parser.TSVFileParser;
import com.example.javawork.Database.DatabaseManager;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class FileSelectorController {

    @FXML
    private Button Browsefile;

    @FXML
    private Button CreateTable;

    @FXML
    private TextField filePathField;

    @FXML
    private TextField tableNameField;

    @FXML
    private Label messageLabel;

    private Stage primaryStage;

    public void setPrimaryStage(Stage primaryStage) {
        this.primaryStage = primaryStage;
    }

    @FXML
    private void browseFile() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select TSV File");

        // 添加文件扩展名过滤器，只允许选择 TSV 文件
        FileChooser.ExtensionFilter extFilter = new FileChooser.ExtensionFilter("TSV files (*.tsv)", "*.tsv");
        fileChooser.getExtensionFilters().add(extFilter);

        File selectedFile = fileChooser.showOpenDialog(primaryStage);
        if (selectedFile != null) {
            // 设置路径到文本框
            filePathField.setText(selectedFile.getAbsolutePath());
        }
    }

    @FXML
    private void createTable() {
        String filePath = filePathField.getText();
        String tableName = tableNameField.getText();

        if (!isValidTableName(tableName)) {
            // 若表名不合法，弹出窗口要求重新输入
            showMessage("Invalid table name. Please enter a valid table name.");
            return;
        }

        DatabaseManager databaseManager = new DatabaseManager();
        if (isTableExists(tableName, databaseManager)) {
            showMessage("Table already exists. Please use another name.");
            databaseManager.closeConnection();
        }else {
            TSVFileParser tsvFileParser = new TSVFileParser(databaseManager);
            tsvFileParser.parserData(filePath, tableName);

            // For demonstration purposes, show success message
            showMessage("Table created successfully.");
        }
    }

    private boolean isTableExists(String tableName, DatabaseManager databaseManager) {
        boolean tableExists = false;
        if (databaseManager != null) {
            try {
                tableExists = databaseManager.isTableExists(tableName);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return tableExists;
    }

    private void showMessage(String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Tip");
        alert.setHeaderText(null);
        alert.setContentText(message);

        alert.showAndWait();
    }

    private boolean isTSVFile(String filePath) {
        File file = new File(filePath);
        String fileName = file.getName();
        String extension = fileName.substring(fileName.lastIndexOf(".") + 1);

        return extension.equalsIgnoreCase("tsv") || extension.equalsIgnoreCase("txt");
    }

    private boolean isValidTableName(String tableName) {
        // 检查表名是否为空
        if (tableName == null || tableName.isEmpty()) {
            return false;
        }

        // 检查表名长度是否超过限制
        if (tableName.length() > 64) {
            return false;
        }

        // 检查表名是否以数字开头
        if (Character.isDigit(tableName.charAt(0))) {
            return false;
        }

        // 检查表名中的字符是否合法
        for (char c : tableName.toCharArray()) {
            if (!Character.isLetterOrDigit(c) && c != '_') {
                return false;
            }
        }

        // 检查表名是否是 SQLite 保留关键字
        List<String> reservedKeywords = Arrays.asList(
                "ABORT", "ACTION", "ADD"/* ... other SQLite keywords ... */);
        if (reservedKeywords.contains(tableName.toUpperCase())) {
            return false;
        }

        return true;
    }

}

