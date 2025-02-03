package com.example.javawork.Search;

import com.example.javawork.DataBrowser.DataBrowserController;
import com.example.javawork.Database.DatabaseManager;
import com.example.javawork.ShowSearchData.ShowSearchDataApp;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.TextField;
import javafx.stage.Stage;

import java.io.File;
import java.sql.SQLException;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class SearchController {

    @FXML
    private Button Search;

    @FXML
    private ChoiceBox<String> KeywordChoiceBox;

    @FXML
    private TextField SerachTerm;

    @FXML
    private CheckBox exactMatch;

    private String tableName = DataBrowserController.tableName;

    public static String columnName;

    public static String[]terms;

    public static Boolean flag = false;

    private int pageSize = 25; // 每页数据量

    private Queue<Integer> rownumber = new LinkedList<>();



    @FXML
    void exactMatch(){

    }
    @FXML
    void Search() {

        String str = SerachTerm.getText();
        terms = str.split(" ");
        exactMatch.selectedProperty().addListener((observable, oldValue, newValue) -> {
            flag = newValue; // 设置 flag 的值为 CheckBox 的新状态
        });
        DatabaseManager.matchingData.clear();//返回数据初始化
        Stage newStage = new Stage();
        ShowSearchDataApp showSearchDataApp = new ShowSearchDataApp();
        showSearchDataApp.start(newStage);
        //System.out.println(flag);
    }

    @FXML
    private void initialize() {
        try {
            // 从数据库获取的列名
            DatabaseManager databaseManager = new DatabaseManager();
            List<String> databaseList = databaseManager.getTableColumnNames(tableName);
            KeywordChoiceBox.getItems().addAll(databaseList);
            System.out.println(databaseList.size());
            // 设置选择监听器，当用户选择列名时触发
            KeywordChoiceBox.setOnAction(event -> {
                String selectedColumn = KeywordChoiceBox.getValue();
                // 在这里执行用户选择列名后的操作
                System.out.println("Selected Database column: " + selectedColumn);
                // 将列名传递给其他类
                columnName = selectedColumn;
            });
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }


}
