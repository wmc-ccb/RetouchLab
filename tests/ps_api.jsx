// Query Photoshop's real selection API surface.
// Run via: Photoshop.exe -r script.jsx  (or via COM/VBScript)
#target photoshop

function main() {
    var out = [];
    out.push("PS version: " + app.version);

    // 1) 选择主体：脚本 API 只有 selectSubject，没有分类参数
    out.push("selectSubject exists: " + (typeof (function () {
        try {
            var d = new ActionDescriptor();
            d.putClass(charIDToTypeID("null"), charIDToTypeID("Mn  "));
            var r = new ActionReference();
            r.putProperty(charIDToTypeID("Prpr"), stringIDToTypeID("selection"));
            r.putEnumerated(charIDToTypeID("Dcmn"), charIDToTypeID("Ordn"),
                            charIDToTypeID("Trgt"));
            d.putReference(charIDToTypeID("null"), r);
            var d2 = new ActionDescriptor();
            d2.putBoolean(stringIDToTypeID("sampleAllLayers"), false);
            d.putObject(charIDToTypeID("T   "),
                        charIDToTypeID("Msk "), d2);
            return true;
        } catch (e) {
            return "err:" + e.message;
        }
    })()));

    // 2) 列出「选择」菜单的真实命令 ID（通过菜单回调不可枚举，
    //    改为查询是否存在"选择人物"相关的 stringID）
    var probes = [
        "selectPeople", "selectPerson", "selectSubjectDetailed",
        "selectSkin", "selectClothes", "selectHair", "selectFace",
        "selectBody", "selectBackground"
    ];
    for (var i = 0; i < probes.length; i++) {
        try {
            var id = stringIDToTypeID(probes[i]);
            out.push(probes[i] + " => id=" + id);
        } catch (e) {
            out.push(probes[i] + " => NOT FOUND");
        }
    }

    var f = new File("C:/Users/Rana/ZhaJinWenQianCake/out/ps_api.txt");
    f.encoding = "UTF8";
    f.open("w");
    f.write(out.join("\n"));
    f.close();
}

try {
    main();
} catch (e) {
    var f2 = new File("C:/Users/Rana/ZhaJinWenQianCake/out/ps_api_err.txt");
    f2.open("w");
    f2.write(e.message + "\n" + e.line);
    f2.close();
}
