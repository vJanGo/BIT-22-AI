package com.example.javawork.DataBrowser;
import com.example.javawork.Search.SearchApp;
import javafx.event.ActionEvent;
import javafx.scene.control.ChoiceBox;
import com.example.javawork.FileSelector.FileSelectorApp;
import com.example.javawork.ShowData.ShowDataApp;
import com.example.javawork.Database.DatabaseManager;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.TableView;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;
import java.io.IOException;
import java.net.URL;
import java.sql.SQLException;
import java.util.List;
import java.util.ResourceBundle;

import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Pagination;

public class DataBrowserController {

    @FXML
    private ChoiceBox<String> databaseChoiceBox;

    public static String tableName;

    @FXML
    private Button DeriveData;

    @FXML
    private Pagination Pages;

    @FXML
    private Button SearchData;

    @FXML
    private Button AnalyzeData;

    @FXML
    private TableView<List<String>> dataTable; // 表格控件，用于展示数据

    @FXML
    private Pagination pagination; // 用于翻页

    @FXML
    void ShowData() {

        Stage stage = new Stage();
        ShowDataApp showDataApp = new ShowDataApp();
        showDataApp.start(stage);

    }

    @FXML
    private void openFileSelector() throws IOException {

        Stage newStage = new Stage();
        FileSelectorApp fileSelectorApp = new FileSelectorApp();
        fileSelectorApp.startFileSelector(newStage);
    }




    @FXML
    void SearchData() throws IOException {

        Stage newStage = new Stage();
        SearchApp searchApp = new SearchApp();
        searchApp.start(newStage);

    }

    @FXML
    void AnalyzeData() {

    }

    @FXML
    private void initialize()
    {
        updateDatabaseChoiceBox();
    }
    @FXML
    private void onChoiceBoxClicked(MouseEvent event) {
        updateDatabaseChoiceBox(); // 在用户点击时更新下拉菜单的库名列表
    }

    private void updateDatabaseChoiceBox() {
        try {
            // 从数据库获取最新的库名列表
            DatabaseManager databaseManager = new DatabaseManager();
            List<String> updatedDatabaseList = databaseManager.getDatabaseNames();

            // 清空原有下拉菜单内容并更新
            databaseChoiceBox.getItems().clear();
            databaseChoiceBox.getItems().addAll(updatedDatabaseList);
            // 设置选择监听器，当用户选择库名时触发
            databaseChoiceBox.setOnAction(event -> {
                String selectedDatabase = databaseChoiceBox.getValue();
                // 在这里执行用户选择库名后的操作
                System.out.println("Selected Database: " + selectedDatabase);
                // 将库名传递给其他类或执行其他操作
                tableName = selectedDatabase;
            });

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

}
