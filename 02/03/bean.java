Connection c = driver.getConnection();
try {
    ...
} finally {
    c.close();
}
