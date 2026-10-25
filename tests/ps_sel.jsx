// Record what "Select Subject" actually does, by executing it and dumping
// the resulting selection state. Also probe the "Select and Mask" workspace
// modes which expose the per-part selection.
#target photoshop

function dumpSelection() {
    var out = [];
    try {
        var r = new ActionReference();
        r.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("fsel"));
        r.putEnumerated(charIDToTypeID("Dcmn"), charIDToTypeID("Ordn"),
                        charIDToTypeID("Trgt"));
        var d = executeActionGet(r);
        if (!d.hasKey(charIDToTypeID("fsel"))) {
            return "no selection";
        }
        var sel = d.getObjectValue(charIDToTypeID("fsel"));
        out.push("has selection: true");
        if (sel.hasKey(charIDToTypeID("Bnds"))) {
            var b = sel.getObjectValue(charIDToTypeID("Bnds"));
            var l = b.getUnitDoubleValue(charIDToTypeID("Left"));
            var t = b.getUnitDoubleValue(charIDToTypeID("Top "));
            var rr = b.getUnitDoubleValue(charIDToTypeID("Rght"));
            var bb = b.getUnitDoubleValue(charIDToTypeID("Btom"));
            out.push("bounds: " + Math.round(l) + "," + Math.round(t) +
                     " " + Math.round(rr) + "," + Math.round(bb));
        }
    } catch (e) {
        out.push("dump err: " + e.message);
    }
    return out.join("\n");
}

function main() {
    var out = [];
    out.push("doc: " + app.activeDocument.name);
    out.push("--- before ---");
    out.push(dumpSelection());

    // 执行「选择主体」
    try {
        var d = new ActionDescriptor();
        var r = new ActionReference();
        r.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("fsel"));
        r.putEnumerated(charIDToTypeID("Dcmn"), charIDToTypeID("Ordn"),
                        charIDToTypeID("Trgt"));
        d.putReference(charIDToTypeID("null"), r);
        var d2 = new ActionDescriptor();
        d2.putBoolean(stringIDToTypeID("sampleAllLayers"), false);
        d.putObject(charIDToTypeID("T   "), charIDToTypeID("Msk "), d2);
        executeAction(charIDToTypeID("setd"), d, DialogModes.NO);
        out.push("--- after selectSubject ---");
        out.push(dumpSelection());
    } catch (e) {
        out.push("selectSubject err: " + e.message);
    }

    // 检查是否存在"选择并遮住"里的分部件选择命令
    // 用 ActionDescriptor 探测 PS 认识哪些 selection 相关命令
    var probes = [
        ["selectSubjectDetail", "选择主体详细"],
        ["selectPPeople", "选择人物"],
        ["selectPersonV2", "选择人物V2"]
    ];
    out.push("--- probe action names ---");
    for (var i = 0; i < probes.length; i++) {
        var name = probes[i][0];
        try {
            var cid = charIDToTypeID(name);
            out.push(name + " charID ok: " + cid);
        } catch (e) {
            out.push(name + " charID NOT a 4-char code");
        }
    }

    var f = new File("C:/Users/Rana/ZhaJinWenQianCake/out/ps_sel.txt");
    f.encoding = "UTF8";
    f.open("w");
    f.write(out.join("\n"));
    f.close();
}

try { main(); } catch (e) {
    var f2 = new File("C:/Users/Rana/ZhaJinWenQianCake/out/ps_sel_err.txt");
    f2.open("w"); f2.write(e.message); f2.close();
}
