/* COLEMAN-authored JUnit3 observation harness, distributed under the repository MIT license. */
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TimeZone;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestFailure;
import junit.framework.TestResult;
import junit.framework.TestSuite;

/** Observe a pinned common test tree; success means collection, never an all-green suite. */
public final class ColemanJunit3Capture {
    private static final int DIAGNOSTIC_BYTES = 65536;

    private static final class LimitedBytes extends OutputStream {
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        private boolean truncated;
        @Override public void write(int value) {
            if (bytes.size() < DIAGNOSTIC_BYTES) bytes.write(value); else truncated = true;
        }
        @Override public void write(byte[] value, int offset, int length) {
            int accepted = Math.min(length, DIAGNOSTIC_BYTES - bytes.size());
            bytes.write(value, offset, accepted);
            if (accepted != length) truncated = true;
        }
        private String text() { return new String(bytes.toByteArray(), StandardCharsets.UTF_8); }
    }

    private static String quote(String value) {
        StringBuilder out = new StringBuilder("\"");
        for (char c : value.toCharArray()) {
            switch (c) {
                case '\\': out.append("\\\\"); break;
                case '"': out.append("\\\""); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                default:
                    if (c < 32) out.append(String.format("\\u%04x", (int)c)); else out.append(c);
            }
        }
        return out.append('"').toString();
    }

    private static String array(List<String> values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i != 0) out.append(',');
            out.append(quote(values.get(i)));
        }
        return out.append(']').toString();
    }

    private static String identity(TestCase test) {
        return test.getClass().getName() + "::" + test.getName();
    }

    @SuppressWarnings("unchecked")
    private static List<TestCase> discover(List<String> excluded) throws Exception {
        List<TestCase> cases = new ArrayList<>();
        for (String name : Files.readAllLines(Paths.get("/harness/classes.txt"), StandardCharsets.UTF_8)) {
            Class<?> type = Class.forName(name);
            if (!TestCase.class.isAssignableFrom(type) || Modifier.isAbstract(type.getModifiers())) {
                excluded.add(name);
                continue;
            }
            TestSuite suite = new TestSuite((Class<? extends TestCase>)type);
            Enumeration<Test> tests = suite.tests();
            while (tests.hasMoreElements()) {
                Test test = tests.nextElement();
                if (!(test instanceof TestCase)) throw new IllegalStateException("Unexpected nested suite");
                cases.add((TestCase)test);
                if (cases.size() > 10000) throw new IllegalStateException("Discovery exceeds case limit");
            }
        }
        cases.sort(Comparator.comparing(ColemanJunit3Capture::identity));
        Set<String> identities = new HashSet<>();
        for (TestCase test : cases) {
            if (!identities.add(identity(test))) throw new IllegalStateException("Duplicate discovered identity");
        }
        excluded.sort(String::compareTo);
        return cases;
    }

    private static void observe(TestCase test, PrintStream output, PrintStream error) throws Exception {
        LimitedBytes diagnostics = new LimitedBytes();
        TestResult result = new TestResult();
        String started = Instant.now().truncatedTo(ChronoUnit.MICROS).toString();
        long begin = System.nanoTime();
        try (PrintStream capture = new PrintStream(diagnostics, true, "UTF-8")) {
            System.setOut(capture);
            System.setErr(capture);
            try { test.run(result); }
            finally { System.setOut(output); System.setErr(error); }
            long elapsed = System.nanoTime() - begin;
            String finished = Instant.now().truncatedTo(ChronoUnit.MICROS).toString();
            for (Enumeration<TestFailure> e = result.failures(); e.hasMoreElements();) capture.print(e.nextElement().trace());
            for (Enumeration<TestFailure> e = result.errors(); e.hasMoreElements();) capture.print(e.nextElement().trace());
            capture.flush();
            output.println("{\"schema_version\":1,\"record_type\":\"case\",\"test_id\":" + quote(identity(test))
                + ",\"tests_run\":" + result.runCount() + ",\"failures\":" + result.failureCount()
                + ",\"errors\":" + result.errorCount() + ",\"elapsed_ns\":" + elapsed
                + ",\"capture_started_at\":" + quote(started) + ",\"capture_finished_at\":" + quote(finished)
                + ",\"diagnostics\":" + quote(diagnostics.text())
                + ",\"diagnostics_truncated\":" + diagnostics.truncated + "}");
        }
    }

    /** Emit one header, every lexical shared-JVM case and a completion trailer as UTF-8 JSON lines. */
    public static void main(String[] args) throws Exception {
        PrintStream output = System.out;
        PrintStream error = System.err;
        List<String> excluded = new ArrayList<>();
        List<TestCase> cases = discover(excluded);
        List<String> identities = new ArrayList<>();
        for (TestCase test : cases) identities.add(identity(test));
        output.println("{\"schema_version\":1,\"record_type\":\"header\",\"java_version\":"
            + quote(System.getProperty("java.version")) + ",\"architecture\":" + quote(System.getProperty("os.arch"))
            + ",\"timezone\":" + quote(TimeZone.getDefault().getID())
            + ",\"capture_clock\":\"producer-truncated-utc-microsecond-v1\""
            + ",\"test_ids\":" + array(identities) + ",\"excluded_classes\":" + array(excluded) + "}");
        for (TestCase test : cases) observe(test, output, error);
        output.println("{\"schema_version\":1,\"record_type\":\"trailer\",\"case_count\":" + cases.size() + "}");
    }
}
