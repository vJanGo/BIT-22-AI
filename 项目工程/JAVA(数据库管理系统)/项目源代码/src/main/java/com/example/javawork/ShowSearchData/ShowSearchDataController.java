package com.example.javawork.ShowSearchData;

import com.example.javawork.Search.SearchController;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.stage.FileChooser;

import java.io.File;
import java.io.IOException;
import java.io.FileWriter;
import com.example.javawork.DataBrowser.DataBrowserController;
import com.example.javawork.Database.DatabaseManager;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.control.cell.CheckBoxTableCell;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import javafx.util.Callback;

import java.net.URL;
import java.sql.SQLException;
import java.util.*;

public class ShowSearchDataController implements Initializable {

    @FXML
    private Pagination pagination;

    @FXML
    private TableView<String[]> dataTable;

    @FXML
    private TableColumn<String[], Boolean> selectionColumn;

    @FXML
    private TableColumn<String[], String> column1 ;

    @FXML
    private TableColumn<String[], String> column2;

    @FXML
    private TableColumn<String[], String> column3;

    @FXML
    private TableColumn<String[], String> column4;

    @FXML
    private TableColumn<String[], String> column5;

    @FXML
    private TableColumn<String[], String> column6;

    @FXML
    private TableColumn<String[], String> column7;

    @FXML
    private GridPane gridPane;

    @FXML
    private TextField pageInputField;


    private String tableName = DataBrowserController.tableName;

    private String[]terms = SearchController.terms;

    private Boolean exactMatch = SearchController.flag;

    private String columnName = SearchController.columnName;
    private int pageSize = 25; // 每页数据量

    private Queue<Integer> rownumber = new LinkedList<>();

    private File selectedFile;
    DatabaseManager databaseManager = new DatabaseManager();

    @Override
    public void initialize(URL url, ResourceBundle rb) {
        if(tableName == null)
        {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Database Empty");
            alert.setHeaderText("The database is empty.");
            alert.setContentText("Please select one database.");
            alert.showAndWait();
            return;
        }
        try {
            displayTableData();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        // Initialize the selection column

        selectionColumn.setCellValueFactory(cellData -> {
            String[] row = cellData.getValue();
            SimpleBooleanProperty observable = new SimpleBooleanProperty(Boolean.parseBoolean(row[0]));
            observable.addListener((obs, oldValue, newValue) -> {
                int rowIndex = cellData.getTableView().getItems().indexOf(row);
                if (newValue) {
                    if (!rownumber.contains(rowIndex)) {
                        rownumber.offer(rowIndex);
                    }
                } else {
                    rownumber.remove(rowIndex);
                }
            });
            return observable;
        });

        // Add CheckBoxTableCell to selectionColumn
        selectionColumn.setCellFactory(column -> new CheckBoxTableCell<>());
    }
    private void displayTableData() throws SQLException {
        List<String> columnNames = determineColumnCount(); // 获取数据的列数

        for (int i = 0; i < columnNames.size(); i++) {
            int columnIndex = i;
            TableColumn<String[], String> column = new TableColumn<>(columnNames.get(i));
            column.setCellValueFactory(cellData -> {
                String[] row = cellData.getValue();
                if (columnIndex < row.length) {
                    return new SimpleStringProperty(row[columnIndex]);
                } else {
                    return new SimpleStringProperty(""); // 如果超出了数组长度，返回空字符串
                }
            });
            dataTable.getColumns().add(column);
        }

        databaseManager.searchDataByColumn(terms,columnName,tableName,exactMatch);//查询初始化
        // 设置数据显示
        int totalRecords = databaseManager.getSearchDataRowCount(); // 查询数据库获得总行数
        int totalPages = (int) Math.ceil((double) totalRecords / pageSize);
        pagination.setPageCount(totalPages);
        pagination.setPageFactory(new Callback<Integer, Node>() {
            @Override
            public Node call(Integer param) {
                return loadPageData(param + 1);
            }
        });
        dataTable.setEditable(true);
    }
    private List<String> determineColumnCount() throws SQLException {
        List<String> columnNames = databaseManager.getTableColumnNames(tableName);
        return columnNames;
    }
    private ObservableList<String[]> getPageData(int pageIndex) {
        List<List<String>> pageData = databaseManager.GetSearchData(pageIndex, pageSize);
        ObservableList<String[]> observablePageData = FXCollections.observableArrayList();

        for (List<String> data : pageData) {
            observablePageData.add(data.toArray(new String[0]));
        }

        return observablePageData;
    }

    private Node loadPageData(int pageIndex) {
        dataTable.getItems().clear();
        dataTable.setItems(getPageData(pageIndex));
        return dataTable;
    }

    @FXML
    private void gotoonepage() {
        // 在这里处理跳转到某一页的逻辑
        // 从 pageInputField 中获取用户输入的页数，然后执行相应的操作
        String inputText = pageInputField.getText();
        int pageNumber = Integer.parseInt(inputText); // 将输入的文本转换为整数页数
        pagination.setCurrentPageIndex(pageNumber - 1); // 设置Pagination控件显示指定页
        loadPageData(pageNumber); // 加载对应页数的数据到TableView中
    }
    @FXML
    private void Download() throws SQLException {
        if (selectedFile == null) {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("File Empty");
            alert.setHeaderText("The selected file is empty.");
            alert.setContentText("Please select another file.");
            alert.showAndWait();
            return;
        }
        ObservableList<String[]> selectedRows = FXCollections.observableArrayList();

        for (Integer rowIndex : rownumber) {
            selectedRows.add(dataTable.getItems().get(rowIndex));
        }
        /*while (!rownumber.isEmpty()){
            System.out.println(rownumber.remove());
        }*/
        if (!selectedRows.isEmpty()) {
            // Create a StringBuilder to hold the data
            StringBuilder data = new StringBuilder();

            // Append selected rows data (excluding the first column)
            for (String[] row : selectedRows) {
                for (int i = 0; i < row.length; i++) {
                    data.append(row[i]).append("\t");
                }
                data.deleteCharAt(data.length() - 1); // Remove trailing comma
                data.append("\n");
            }

            // Save data to a file
            try {
                FileWriter fileWriter = new FileWriter(selectedFile,true);
                fileWriter.write(data.toString());
                fileWriter.close();
                System.out.println("Selected data has been saved to selected_data.tsv");
                Alert alert = new Alert(Alert.AlertType.INFORMATION);
                alert.setTitle("File Saved");
                alert.setHeaderText("The data is saved.");
                alert.setContentText("Please select another data or exit.");
                alert.showAndWait();
                rownumber.clear();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("No rows selected.");
        }
    }
    @FXML
    private void openFileChooser() throws SQLException, IOException {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save as TSV File");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("TSV Files", "*.tsv"));

        // Show save file dialog
        Stage stage = (Stage) dataTable.getScene().getWindow(); // Get the reference to the current stage
        selectedFile = fileChooser.showSaveDialog(stage);

        if (selectedFile != null) {
            StringBuilder data = new StringBuilder();
            List<String> ColumnNames = databaseManager.getTableColumnNames(tableName);
            for (String ColoumnName : ColumnNames)
            {
                data.append(ColoumnName).append("\t");
            }
            data.append("\n");
            FileWriter fileWriter = new FileWriter(selectedFile,true);
            fileWriter.write(data.toString());
            fileWriter.close();
        }
    }

    @FXML
    private void BatchDownload() {
        // Create a dialog for user input
        Dialog<List<Integer>> dialog = new Dialog<>();
        dialog.setTitle("Download Pages");
        dialog.setHeaderText("Enter start and end page numbers");

        // Set the button types
        ButtonType downloadButtonType = new ButtonType("Download", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(downloadButtonType, ButtonType.CANCEL);

        // Create and configure the GridPane for the content
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20, 150, 10, 10));

        // Create input fields for start and end page numbers
        TextField startPageField = new TextField();
        startPageField.setPromptText("Start Page");
        TextField endPageField = new TextField();
        endPageField.setPromptText("End Page");

        grid.add(new Label("Start Page:"), 0, 0);
        grid.add(startPageField, 1, 0);
        grid.add(new Label("End Page:"), 0, 1);
        grid.add(endPageField, 1, 1);

        // Enable/disable download button depending on input validation
        Node downloadButton = dialog.getDialogPane().lookupButton(downloadButtonType);
        downloadButton.setDisable(true);

        startPageField.textProperty().addListener((observable, oldValue, newValue) -> {
            downloadButton.setDisable(newValue.trim().isEmpty() || endPageField.getText().trim().isEmpty() || !newValue.matches("\\d+"));
        });

        endPageField.textProperty().addListener((observable, oldValue, newValue) -> {
            downloadButton.setDisable(newValue.trim().isEmpty() || startPageField.getText().trim().isEmpty() || !newValue.matches("\\d+"));
        });

        dialog.getDialogPane().setContent(grid);

        // Request focus on the startPageField by default
        Platform.runLater(startPageField::requestFocus);

        // Convert the result to a List<Integer> when the download button is clicked
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == downloadButtonType) {
                List<Integer> result = new ArrayList<>();
                result.add(Integer.parseInt(startPageField.getText()));
                result.add(Integer.parseInt(endPageField.getText()));
                return result;
            }
            return null;
        });

        dialog.showAndWait().ifPresent(pages -> {
            int startPage = pages.get(0);
            int endPage = pages.get(1);

            if (startPage >= 1 && endPage >= startPage) {
                // Call a method to download pages from startPage to endPage
                downloadPages(startPage, endPage);
            } else {
                // Show an error message if the input is invalid
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("Invalid Input");
                alert.setHeaderText("Invalid page range");
                alert.setContentText("Please enter valid page numbers.");
                alert.showAndWait();
            }
        });
    }

    // Method to download pages from startPage to endPage
    private void downloadPages(int startPage, int endPage) {
        try {
            if (selectedFile == null) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("File Empty");
                alert.setHeaderText("The selected file is empty.");
                alert.setContentText("Please select another file.");
                alert.showAndWait();
                return;
            }
            // Create a FileWriter for writing data to the file
            FileWriter fileWriter = new FileWriter(selectedFile);

            // Write column headers to the file
            List<String> columnNames = databaseManager.getTableColumnNames(tableName);
            for (String columnName : columnNames) {
                fileWriter.write(columnName + "\t");
            }
            fileWriter.write("\n");

            // Download pages from startPage to endPage
            for (int pageIndex = startPage; pageIndex <= endPage; pageIndex++) {
                ObservableList<String[]> pageData = getPageData(pageIndex);

                // Write each row of data to the file
                for (String[] row : pageData) {
                    for (String cell : row) {
                        fileWriter.write(cell + "\t");
                    }
                    fileWriter.write("\n");
                }
            }

            // Close the FileWriter
            fileWriter.close();
            System.out.println("Selected pages have been saved to selected_pages.tsv");
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("File Saved");
            alert.setHeaderText("The data is saved.");
            alert.setContentText("Please select another data or exit.");
            alert.showAndWait();
        } catch (IOException | SQLException e) {
            e.printStackTrace();
        }
    }


}
