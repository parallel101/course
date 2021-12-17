Connection c = driver.getConnection();
try {
    ...
} catch (SQLException e) {
    ...
} finally {
    c.close();
}
